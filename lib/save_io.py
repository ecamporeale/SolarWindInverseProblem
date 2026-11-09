from pathlib import Path


def _save_batch_summary_and_loss_plot(all_solutions, output_dir):
    """Save a simple summary file for parallel batch execution."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "batch_summary.txt"
    lines = [
        "Parallel batch execution summary:\n",
        f"Total batches: {len(all_solutions.get('batches', []))}\n",
    ]
    for batch in all_solutions.get('batches', []):
        lines.append(
            f"batch_idx={batch.get('batch_idx')} final_loss={batch.get('final_loss')}\n"
        )
    summary_path.write_text(''.join(lines))
