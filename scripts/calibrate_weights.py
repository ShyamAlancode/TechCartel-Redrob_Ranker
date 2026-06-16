"""
Calibration script for finding optimal weights (Phase 4).
"""
import json
from pathlib import Path
import sys
import numpy as np

# Adjust path so we can import from ranker
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ranker import config
from ranker.loading import load_candidates_blob
from ranker.pipeline import select_top

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/calibrate_weights.py <path_to_golden_set.json>")
        return

    golden_path = Path(sys.argv[1])
    if not golden_path.exists():
        print(f"File {golden_path} not found.")
        return
        
    with open(golden_path) as f:
        golden = json.load(f)
    
    perfect_ids = set(golden.get("perfect", []))
    trap_ids = set(golden.get("traps", []))
    
    print(f"Loaded {len(perfect_ids)} perfect and {len(trap_ids)} trap candidates.")
    
    # Load all candidates from sample
    sample_path = Path("official docs/sample_candidates.json")
    if not sample_path.exists():
        print(f"Sample candidates not found at {sample_path}.")
        return
        
    with open(sample_path, "rb") as f:
        all_candidates = load_candidates_blob(f.read())
        
    artifacts = Path("artifacts")
    if not (artifacts / "candidate_embeddings.npy").exists():
        print("Embeddings not found. Run embed.py first.")
        return
        
    # Replace dummy random embeddings with actual loaded embeddings
    embeddings = np.load(artifacts / "candidate_embeddings.npy")
    jd_vec = np.load(artifacts / "jd_embedding.npy")
    sims = embeddings @ jd_vec
    lo, hi = sims.min(), sims.max()
    span = (hi - lo) or 1.0
    semantic_lookup_dict = {c["candidate_id"]: float((s - lo) / span) for c, s in zip(all_candidates, sims)}
    
    best_loss = float('inf')
    best_weights = None
    
    weights_sem = [0.3, 0.4, 0.5, 0.6, 0.7]
    weights_struc = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\nStarting Grid Search...")
    for w_sem in weights_sem:
        for w_str in weights_struc:
            # Skip if they don't roughly sum to 1.0 (with a bit of margin)
            if abs(w_sem + w_str - 1.0) > 0.1:
                continue
                
            # Temporarily patch config
            config.W_SEMANTIC = w_sem
            config.W_STRUCTURAL = w_str
            
            # Score
            ranked = select_top(all_candidates, semantic_lookup_dict.get, top_k=len(all_candidates))
            
            # Calculate loss: 
            # Perfect candidates should be at rank 0, Trap candidates should be >= 100
            loss = 0
            for i, cand in enumerate(ranked):
                if cand.candidate_id in perfect_ids:
                    loss += i
                elif cand.candidate_id in trap_ids:
                    loss += max(0, 100 - i)
                    
            print(f"Semantic={w_sem:.1f}, Structural={w_str:.1f} -> Loss: {loss}")
            
            if loss < best_loss:
                best_loss = loss
                best_weights = (w_sem, w_str)
                
    print("\n=============================================")
    print(f"Best Weights: Semantic={best_weights[0]:.1f}, Structural={best_weights[1]:.1f}")
    print(f"Minimum Loss: {best_loss}")
    print("=============================================")

if __name__ == "__main__":
    main()
