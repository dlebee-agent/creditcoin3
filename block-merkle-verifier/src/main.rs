//! block-merkle-verifier
//!
//! Given a block height, an expected Keccak256 Merkle root, and an Ethereum-compatible
//! RPC URL, this tool:
//!
//!   1. Fetches the full block (with transactions) via `eth_getBlockByNumber`.
//!   2. Fetches all receipts for that block via `eth_getBlockReceipts`.
//!      Both calls go to the **same** RPC provider so a reorg between calls is
//!      detected immediately (header root mismatch → error).
//!   3. ABI-encodes each (tx, receipt) pair using the same V1 encoding used by
//!      the creditcoin3 attestation pipeline.
//!   4. Builds a Keccak256 Merkle tree over those encoded leaves.
//!   5. Compares the computed root to the expected root you supplied.
//!
//! # Usage
//!
//! ```text
//! block-merkle-verifier \
//!     --height 7654321 \
//!     --root 0xabc123... \
//!     --rpc https://rpc.example.com
//! ```

use alloy::primitives::{keccak256, B256};
use anyhow::{Context, Result};
use clap::Parser;
use eth::Client;
use user::prelude::Interrupt;
use utils::block_item_traits::BlockItem;

// ─── Merkle root constants (mirror creditcoin3/common/merkle/src/keccak.rs) ──
const LEAF_PREPEND: u8 = 0;
const INNER_PREPEND: u8 = 1;

// ─── CLI ─────────────────────────────────────────────────────────────────────

/// Fetch a block, encode its transactions + receipts, compute the Keccak256
/// Merkle root, and compare it to an expected root.
#[derive(Parser, Debug)]
#[command(name = "block-merkle-verifier", version, about, long_about = None)]
struct Cli {
    /// Block height (number) to fetch.
    #[arg(long, short = 'n')]
    height: u64,

    /// Expected Merkle root (hex, with or without 0x prefix).
    /// Pass "skip" to skip the comparison and just print the computed root.
    #[arg(long, short = 'r')]
    root: String,

    /// Ethereum-compatible JSON-RPC endpoint (http/https/ws/wss).
    #[arg(long, short = 'u')]
    rpc: String,

    /// ABI encoding version for (tx, receipt) pairs.
    /// Currently only "v1" is supported (matches creditcoin3 attestation pipeline).
    #[arg(long, default_value = "v1")]
    encoding: String,

    /// Show per-leaf hex preview.
    #[arg(long, short = 'v')]
    verbose: bool,
}

// ─── Main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Tracing
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(log_level.parse().unwrap())
                .from_env_lossy(),
        )
        .with_target(false)
        .init();

    // Encoding version
    let encoding = parse_encoding_version(&cli.encoding)?;

    // Expected root
    let expected_root: Option<B256> = if cli.root.eq_ignore_ascii_case("skip") {
        eprintln!("Root comparison skipped (--root skip)");
        None
    } else {
        Some(parse_b256(&cli.root).context("--root must be a 32-byte hex value or \"skip\"")?)
    };

    // ── 1. Connect ───────────────────────────────────────────────────────────
    eprintln!("Connecting to: {}", cli.rpc);
    let client = Client::new(&cli.rpc, None)
        .await
        .context("Failed to connect to RPC endpoint")?;
    eprintln!("Connected  chain_id={}", client.chain_id());

    // ── 2. Fetch block + receipts (sorted, header-verified) ──────────────────
    //
    // `eth::Client::get_block` issues `eth_getBlockByNumber(full=true)` and
    // `eth_getBlockReceipts` concurrently against the **same** provider, then
    // verifies that the recomputed tx-root and receipt-root match the block
    // header.  Mismatches (reorg between the two calls) are returned as errors
    // and will cause us to exit below.
    let block = match client.get_block(cli.height, encoding).await {
        Ok(b) => b,
        Err(Interrupt::Cont(e)) => return Err(e.into()),
        Err(Interrupt::Stop) => {
            eprintln!("Interrupted");
            return Ok(());
        }
    };

    let tx_count = block.items().len();
    eprintln!(
        "Block {}  hash=0x{:x}  txs={}",
        block.number(),
        block.hash(),
        tx_count
    );

    // ── 3. ABI-encode each (tx, receipt) pair ────────────────────────────────
    //
    // `block.items()` returns `TxRx` pairs already sorted by `transaction_index`.
    // `TxRx::to_bytes()` calls `usc_abi_encoding::abi::abi_encode(tx, rx, version)`
    // which is identical to what the attestation pipeline encodes.
    let leaves: Vec<Vec<u8>> = block.items().iter().map(|item| item.to_bytes()).collect();

    if cli.verbose {
        for (i, leaf) in leaves.iter().enumerate() {
            let preview = hex::encode(&leaf[..leaf.len().min(16)]);
            eprintln!("  leaf[{i:04}] len={}  0x{preview}…", leaf.len());
        }
    }

    // ── 4. Compute Keccak256 Merkle root ─────────────────────────────────────
    //
    // Mirrors `common/merkle/src/keccak_merkle_tree.rs` exactly:
    //   • leaf  hash = keccak256(0x00 || item_bytes)
    //   • inner hash = keccak256(0x01 || left || right)
    //   • odd level → missing sibling is the zero hash (H256::default())
    let computed = if leaves.is_empty() {
        B256::ZERO
    } else {
        keccak_merkle_root(&leaves)
    };

    // ── 5. Report ─────────────────────────────────────────────────────────────
    println!();
    println!("block height  : {}", cli.height);
    println!("block hash    : 0x{:x}", block.hash());
    println!("transactions  : {}", tx_count);
    println!("encoding      : {}", cli.encoding);
    println!("computed root : 0x{}", hex::encode(computed.as_slice()));

    if let Some(exp) = expected_root {
        println!("expected root : 0x{}", hex::encode(exp.as_slice()));
    }

    check_root(computed, expected_root);
    Ok(())
}

// ─── Keccak256 Merkle tree ────────────────────────────────────────────────────

/// Compute the Keccak256 Merkle root over `leaves` using the same algorithm
/// as `creditcoin3/common/merkle`.
///
/// * Each item is hashed as `keccak256(0x00 || bytes)`.
/// * Pairs are combined as `keccak256(0x01 || left || right)`.
/// * An odd node at any level is paired with the **zero hash** (not duplicated).
fn keccak_merkle_root(leaves: &[Vec<u8>]) -> B256 {
    // Hash every leaf
    let mut level: Vec<B256> = leaves
        .iter()
        .map(|item| {
            let mut buf = Vec::with_capacity(1 + item.len());
            buf.push(LEAF_PREPEND);
            buf.extend_from_slice(item);
            keccak256(&buf)
        })
        .collect();

    // Walk up the tree until a single root remains
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        let mut i = 0;
        while i < level.len() {
            let left = level[i];
            let right = if i + 1 < level.len() {
                level[i + 1]
            } else {
                B256::ZERO // zero-pad odd node
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
        anyhow::bail!("Expected 32 bytes (64 hex chars) for root, got {}", bytes.len());
    }
    Ok(B256::from_slice(&bytes))
}

fn parse_encoding_version(s: &str) -> Result<usc_abi_encoding::common::EncodingVersion> {
    match s.to_ascii_lowercase().as_str() {
        "v1" | "1" => Ok(usc_abi_encoding::common::EncodingVersion::V1),
        other => anyhow::bail!(
            "Unknown encoding version '{}'. Supported: v1",
            other
        ),
    }
}
