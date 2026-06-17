//! block-merkle-verifier
//!
//! Given a block height (or list of heights), an expected Keccak256 Merkle
//! root, and an Ethereum-compatible RPC URL, this tool:
//!
//!   1. Fetches the full block (with transactions) via `eth_getBlockByNumber`.
//!   2. Fetches all receipts for that block via `eth_getBlockReceipts`.
//!      Both calls go to the **same** RPC provider so a reorg between calls is
//!      detected immediately (header root mismatch → error).
//!   3. ABI-encodes each (tx, receipt) pair using the same V1 encoding used by
//!      the creditcoin3 attestation pipeline.
//!   4. Builds a Keccak256 Merkle tree over those encoded leaves.
//!   5. Compares the computed root to the expected root you supplied
//!      (single-block mode only).
//!
//! # Single-block mode
//!
//! ```text
//! block-merkle-verifier \
//!     --height 7654321 \
//!     --root 0xabc123... \
//!     --rpc https://rpc.example.com
//! ```
//!
//! # Batch mode (up to --concurrency parallel fetches)
//!
//! ```text
//! block-merkle-verifier \
//!     --heights 100,200,300 \
//!     --rpc https://rpc.example.com
//!
//! # Repeated flag also works:
//! block-merkle-verifier \
//!     -H 100 -H 200 -H 300 \
//!     --rpc https://rpc.example.com
//!
//! # Override concurrency (default 10):
//! block-merkle-verifier \
//!     --heights 100,200,300 \
//!     --rpc https://rpc.example.com \
//!     --concurrency 5
//! ```

use alloy::primitives::{keccak256, B256};
use anyhow::{Context, Result};
use clap::Parser;
use eth::{Client, OrderedBlock};
use utils::block_item_traits::BlockItem as _;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use user::prelude::Interrupt;

// ─── Merkle root constants (mirror creditcoin3/common/merkle/src/keccak.rs) ──
const LEAF_PREPEND: u8 = 0;
const INNER_PREPEND: u8 = 1;

// ─── CLI ─────────────────────────────────────────────────────────────────────

/// Fetch one or more blocks, encode their transactions + receipts, compute
/// Keccak256 Merkle roots, and (in single-block mode) compare to an expected
/// root.
#[derive(Parser, Debug)]
#[command(name = "block-merkle-verifier", version, about, long_about = None)]
struct Cli {
    /// Block height (number) to fetch (single-block mode).
    /// Cannot be combined with --heights.
    #[arg(long, short = 'n', conflicts_with = "heights")]
    height: Option<u64>,

    /// One or more block heights for batch mode (comma-separated and/or
    /// repeated). Cannot be combined with --height.
    ///
    /// Examples:
    ///   --heights 100,200,300
    ///   -H 100 -H 200 -H 300
    #[arg(long, short = 'H', value_delimiter = ',', conflicts_with_all = ["height", "input_csv"])]
    heights: Vec<u64>,

    /// CSV file with `block_number,merkle_root` pairs (one per line,
    /// optional header). Each row is verified against its expected root
    /// using the same concurrency cap as batch mode. Cannot be combined
    /// with --height or --heights.
    #[arg(long, short = 'i', conflicts_with_all = ["height", "heights"])]
    input_csv: Option<PathBuf>,

    /// Expected Merkle root (hex, with or without 0x prefix).
    /// Pass "skip" to skip the comparison and just print the computed root.
    /// Ignored in batch mode (always treated as "skip").
    #[arg(long, short = 'r', default_value = "skip")]
    root: String,

    /// Ethereum-compatible JSON-RPC endpoint (http/https/ws/wss).
    #[arg(long, short = 'u')]
    rpc: String,

    /// ABI encoding version for (tx, receipt) pairs.
    /// Currently only "v1" is supported (matches creditcoin3 attestation pipeline).
    #[arg(long, default_value = "v1")]
    encoding: String,

    /// Show per-leaf hex preview (single-block mode only).
    #[arg(long, short = 'v')]
    verbose: bool,

    /// Maximum number of concurrent RPC fetches in batch mode (default: 10).
    #[arg(long, short = 'c', default_value_t = 10)]
    concurrency: usize,
}

// ─── Internal types ──────────────────────────────────────────────────────────────

struct BlockOutput {
    #[allow(dead_code)]
    height: u64,
    block_hash_hex: String,  // pre-formatted "<hex>" without 0x prefix
    tx_count: usize,
    computed_root: B256,
}

/// One input row for batch mode: height + optional expected root.
#[derive(Clone)]
struct BatchJob {
    height: u64,
    expected: Option<B256>,
}

// ─── Main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Determine mode.
    enum Mode {
        Single(u64),
        Batch(Vec<BatchJob>),
    }

    let mode = match (cli.height, cli.heights.is_empty(), cli.input_csv.as_ref()) {
        (Some(h), _, _) => Mode::Single(h),
        (None, false, _) => Mode::Batch(
            cli.heights
                .iter()
                .map(|&height| BatchJob { height, expected: None })
                .collect(),
        ),
        (None, true, Some(path)) => {
            let jobs = parse_csv_jobs(path)
                .with_context(|| format!("reading {}", path.display()))?;
            if jobs.is_empty() {
                eprintln!("error: --input-csv contained no valid rows");
                std::process::exit(2);
            }
            Mode::Batch(jobs)
        }
        (None, true, None) => {
            eprintln!(
                "error: provide --height <N>, --heights <N1,N2,...>, or --input-csv <path>"
            );
            std::process::exit(2);
        }
    };

    // Tracing — in batch mode keep it quieter by default.
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(log_level.parse().unwrap())
                .from_env_lossy(),
        )
        .with_target(false)
        .with_writer(std::io::stderr) // never pollute stdout (CSV output)
        .init();

    let encoding = parse_encoding_version(&cli.encoding)?;

    // Single shared connection; Client is Clone so Arc is optional, but
    // Arc avoids copying the internal connection pool for each task.
    eprintln!("Connecting to: {}", cli.rpc);
    let client = Client::new(&cli.rpc, None)
        .await
        .context("Failed to connect to RPC endpoint")?;
    eprintln!("Connected  chain_id={}", client.chain_id());

    match mode {
        Mode::Single(height) => {
            let expected = if cli.root.eq_ignore_ascii_case("skip") {
                eprintln!("Root comparison skipped (--root skip)");
                None
            } else {
                Some(
                    parse_b256(&cli.root)
                        .context("--root must be a 32-byte hex value or \"skip\"")?,
                )
            };
            run_single(&client, height, encoding, expected, cli.verbose).await
        }
        Mode::Batch(jobs) => {
            if cli.concurrency == 0 {
                anyhow::bail!("--concurrency must be at least 1");
            }
            run_batch(Arc::new(client), jobs, encoding, cli.concurrency).await
        }
    }
}

// ─── Single-block mode ───────────────────────────────────────────────────────

async fn run_single(
    client: &Client,
    height: u64,
    encoding: usc_abi_encoding::common::EncodingVersion,
    expected_root: Option<B256>,
    verbose: bool,
) -> Result<()> {
    let block = fetch_block(client, height, encoding).await?;
    let tx_count = block.items().len();
    let block_hash_hex = format!("{:x}", block.hash());

    eprintln!("Block {}  hash=0x{}  txs={}", block.number(), block_hash_hex, tx_count);

    let leaves: Vec<Vec<u8>> = block.items().iter().map(|item| item.to_bytes()).collect();

    if verbose {
        for (i, leaf) in leaves.iter().enumerate() {
            let preview = hex::encode(&leaf[..leaf.len().min(16)]);
            eprintln!("  leaf[{i:04}] len={}  0x{preview}…", leaf.len());
        }
    }

    let computed = if leaves.is_empty() {
        B256::ZERO
    } else {
        keccak_merkle_root(&leaves)
    };

    println!();
    println!("block height  : {}", height);
    println!("block hash    : 0x{}", block_hash_hex);
    println!("transactions  : {}", tx_count);
    println!("encoding      : v1");
    println!("computed root : 0x{}", hex::encode(computed.as_slice()));

    if let Some(exp) = expected_root {
        println!("expected root : 0x{}", hex::encode(exp.as_slice()));
    }

    check_root(computed, expected_root);
    Ok(())
}

// ─── Batch mode ──────────────────────────────────────────────────────────────

async fn run_batch(
    client: Arc<Client>,
    jobs: Vec<BatchJob>,
    encoding: usc_abi_encoding::common::EncodingVersion,
    concurrency: usize,
) -> Result<()> {
    let total = jobs.len();
    let has_expected = jobs.iter().any(|j| j.expected.is_some());
    eprintln!(
        "Batch: {} blocks  concurrency={}  expected_roots={}",
        total, concurrency, if has_expected { "yes" } else { "no" }
    );

    let sem = Arc::new(Semaphore::new(concurrency));
    let mut set: JoinSet<(BatchJob, Result<BlockOutput>)> = JoinSet::new();

    for job in jobs {
        let client = client.clone();
        let sem = sem.clone();
        set.spawn(async move {
            // Acquire a permit before hitting the RPC.
            let _permit = sem
                .acquire_owned()
                .await
                .expect("semaphore never closes");
            let res = compute_block_root(&client, job.height, encoding).await;
            (job, res)
        });
    }

    // Collect results as they complete; log progress to stderr.
    let mut results: Vec<(BatchJob, Result<BlockOutput>)> = Vec::with_capacity(total);
    let mut done = 0usize;
    let mut matches = 0usize;
    let mut mismatches = 0usize;

    while let Some(join_res) = set.join_next().await {
        done += 1;
        match join_res {
            Ok((job, res)) => {
                match &res {
                    Ok(out) => {
                        let status = match job.expected {
                            Some(exp) if exp == out.computed_root => {
                                matches += 1;
                                "✓ MATCH"
                            }
                            Some(_) => {
                                mismatches += 1;
                                "✗ MISMATCH"
                            }
                            None => "✓ OK",
                        };
                        eprintln!(
                            "  [{done}/{total}] {status} {}  root=0x{}…",
                            job.height,
                            &hex::encode(out.computed_root.as_slice())[..16]
                        );
                    }
                    Err(e) => eprintln!("  [{done}/{total}] ✗ {}  err={e:#}", job.height),
                }
                results.push((job, res));
            }
            Err(join_err) => {
                eprintln!("  [{done}/{total}] task panicked: {join_err}");
            }
        }
    }

    // Sort by height ascending (matches typical input ordering).
    results.sort_by(|a, b| a.0.height.cmp(&b.0.height));

    // Emit CSV to stdout so it can be piped/redirected.
    if has_expected {
        println!("block_number,expected_root,computed_root,block_hash,txns,status");
        for (job, res) in &results {
            let exp_hex = job
                .expected
                .map(|e| format!("0x{}", hex::encode(e.as_slice())))
                .unwrap_or_default();
            match res {
                Ok(out) => {
                    let computed_hex = format!("0x{}", hex::encode(out.computed_root.as_slice()));
                    let status = match job.expected {
                        Some(exp) if exp == out.computed_root => "MATCH",
                        Some(_) => "MISMATCH",
                        None => "OK",
                    };
                    println!(
                        "{},{},{},0x{},{},{}",
                        job.height, exp_hex, computed_hex, out.block_hash_hex, out.tx_count, status
                    );
                }
                Err(e) => println!(
                    "{},{},,,,\"ERROR: {}\"",
                    job.height,
                    exp_hex,
                    e.to_string().replace('"', "'")
                ),
            }
        }
    } else {
        println!("Height,Block Hash,Txns,Computed Merkle Root,Status");
        for (job, res) in &results {
            match res {
                Ok(out) => println!(
                    "{},0x{},{},0x{},OK",
                    job.height,
                    out.block_hash_hex,
                    out.tx_count,
                    hex::encode(out.computed_root.as_slice())
                ),
                Err(e) => println!(
                    "{},,,\"{}\",ERROR",
                    job.height,
                    e.to_string().replace('"', "'")
                ),
            }
        }
    }

    let ok = results.iter().filter(|(_, r)| r.is_ok()).count();
    if has_expected {
        let errors = total - ok;
        eprintln!(
            "\nBatch complete: total={total} match={matches} mismatch={mismatches} error={errors}"
        );
        if mismatches > 0 || errors > 0 {
            std::process::exit(1);
        }
    } else {
        eprintln!("\nBatch complete: {ok}/{total} OK");
        if ok < total {
            std::process::exit(1);
        }
    }

    Ok(())
}

// ─── CSV input parsing ───────────────────────────────────────────────────

/// Parse `block_number,merkle_root` rows. Header is auto-skipped when the
/// first column doesn't parse as an integer. Trailing/extra columns are
/// ignored. Roots may include or omit the `0x` prefix.
fn parse_csv_jobs(path: &std::path::Path) -> Result<Vec<BatchJob>> {
    let text = std::fs::read_to_string(path)?;
    let mut jobs = Vec::new();
    let mut skipped_header = false;
    let mut bad_rows = 0usize;

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(3, ',');
        let h_str = parts.next().unwrap_or("").trim().trim_matches('"');
        let r_str = parts.next().unwrap_or("").trim().trim_matches('"');

        // Skip a header row whose first column isn't an integer.
        let height: u64 = match h_str.parse() {
            Ok(n) => n,
            Err(_) => {
                if !skipped_header && lineno == 0 {
                    skipped_header = true;
                    continue;
                }
                bad_rows += 1;
                eprintln!("  skip: line {} non-numeric height '{}'", lineno + 1, h_str);
                continue;
            }
        };

        let expected = if r_str.is_empty() {
            None
        } else {
            match parse_b256(r_str) {
                Ok(b) => Some(b),
                Err(e) => {
                    bad_rows += 1;
                    eprintln!(
                        "  skip: line {} block {} bad root '{}': {}",
                        lineno + 1,
                        height,
                        r_str,
                        e
                    );
                    continue;
                }
            }
        };
        jobs.push(BatchJob { height, expected });
    }

    if bad_rows > 0 {
        eprintln!("  ({} input rows skipped)", bad_rows);
    }
    Ok(jobs)
}

// ─── Core: fetch block + compute root ───────────────────────────────────────

async fn compute_block_root(
    client: &Client,
    height: u64,
    encoding: usc_abi_encoding::common::EncodingVersion,
) -> Result<BlockOutput> {
    let block = fetch_block(client, height, encoding).await?;
    let tx_count = block.items().len();
    let block_hash_hex = format!("{:x}", block.hash());
    let leaves: Vec<Vec<u8>> = block.items().iter().map(|item| item.to_bytes()).collect();
    let computed_root = if leaves.is_empty() {
        B256::ZERO
    } else {
        keccak_merkle_root(&leaves)
    };
    Ok(BlockOutput {
        height,
        block_hash_hex,
        tx_count,
        computed_root,
    })
}

async fn fetch_block(
    client: &Client,
    height: u64,
    encoding: usc_abi_encoding::common::EncodingVersion,
) -> Result<OrderedBlock> {
    match client.get_block(height, encoding).await {
        Ok(b) => Ok(b),
        Err(Interrupt::Cont(e)) => Err(e.into()),
        Err(Interrupt::Stop) => Err(anyhow::anyhow!("Interrupted")),
    }
}

// ─── Keccak256 Merkle tree ────────────────────────────────────────────────────

/// Compute the Keccak256 Merkle root over `leaves` using the same algorithm
/// as `creditcoin3/common/merkle`.
///
/// * Each item is hashed as `keccak256(0x00 || bytes)`.
/// * Pairs are combined as `keccak256(0x01 || left || right)`.
/// * An odd node at any level is paired with the **zero hash** (not duplicated).
fn keccak_merkle_root(leaves: &[Vec<u8>]) -> B256 {
    let mut level: Vec<B256> = leaves
        .iter()
        .map(|item| {
            let mut buf = Vec::with_capacity(1 + item.len());
            buf.push(LEAF_PREPEND);
            buf.extend_from_slice(item);
            keccak256(&buf)
        })
        .collect();

    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        let mut i = 0;
        while i < level.len() {
            let left = level[i];
            let right = if i + 1 < level.len() {
                level[i + 1]
            } else {
                B256::ZERO
            };
            let mut buf = [INNER_PREPEND; 1 + 64];
            buf[1..33].copy_from_slice(left.as_slice());
            buf[33..65].copy_from_slice(right.as_slice());
            next.push(keccak256(&buf));
            i += 2;
        }
        level = next;
    }

    level.into_iter().next().unwrap_or(B256::ZERO)
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn check_root(computed: B256, expected: Option<B256>) {
    match expected {
        None => {
            println!();
            println!("ℹ️  No expected root supplied — skipping comparison.");
        }
        Some(exp) if computed == exp => {
            println!();
            println!("✅  Merkle root MATCHES.");
        }
        Some(exp) => {
            println!();
            println!("❌  Merkle root MISMATCH.");
            println!("    computed : 0x{}", hex::encode(computed.as_slice()));
            println!("    expected : 0x{}", hex::encode(exp.as_slice()));
            std::process::exit(1);
        }
    }
}

fn parse_b256(s: &str) -> Result<B256> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    let bytes = hex::decode(s).context("Failed to hex-decode root value")?;
    if bytes.len() != 32 {
        anyhow::bail!(
            "Expected 32 bytes (64 hex chars) for root, got {}",
            bytes.len()
        );
    }
    Ok(B256::from_slice(&bytes))
}

fn parse_encoding_version(s: &str) -> Result<usc_abi_encoding::common::EncodingVersion> {
    match s.to_ascii_lowercase().as_str() {
        "v1" | "1" => Ok(usc_abi_encoding::common::EncodingVersion::V1),
        other => anyhow::bail!("Unknown encoding version '{}'. Supported: v1", other),
    }
}
