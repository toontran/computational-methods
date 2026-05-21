import random
import math
from collections import Counter

def f0_estimator(stream, epsilon, delta):
    """
    F₀-Estimator algorithm from the paper
    
    Args:
        stream: list of elements (can have duplicates)
        epsilon: approximation parameter
        delta: confidence parameter
    
    Returns:
        estimate: estimated number of distinct elements
        failed: whether algorithm returned ⊥ (failure)
        final_p: final sampling probability
        final_X_size: final sample size
    """
    # Initialize
    p = 1.0
    X = set()
    thresh = math.ceil(12 / (epsilon**2) * math.log(8 * len(stream) / delta))
    
    for ai in stream:
        # Step 1: Remove ai from X if present
        X.discard(ai)
        
        # Step 2: Add ai to X with probability p
        if random.random() < p:
            X.add(ai)
        
        # Step 3: Check if we need to downsample
        if len(X) == thresh:
            # Throw away each element with probability 1/2
            new_X = set()
            for element in X:
                if random.random() < 0.5:
                    new_X.add(element)
            X = new_X
            p = p / 2
            
            # Check for failure (if still at threshold after downsampling)
            if len(X) == thresh:
                return None, True, p, len(X)  # Return ⊥
    
    # Output |X|/p
    estimate = len(X) / p if p > 0 else 0
    return estimate, False, p, len(X)

def generate_stream(n_distinct, stream_length, zipf_param=1.0):
    """Generate a stream with exactly n_distinct unique elements"""
    # Create n_distinct unique elements
    distinct_elements = list(range(n_distinct))
    
    # Generate stream with repetitions (Zipf distribution for realism)
    stream = []
    weights = [1.0 / (i + 1)**zipf_param for i in range(n_distinct)]
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    
    for _ in range(stream_length):
        # Choose element according to probabilities
        r = random.random()
        cumsum = 0
        for i, prob in enumerate(probabilities):
            cumsum += prob
            if r <= cumsum:
                stream.append(distinct_elements[i])
                break
    
    return stream

def run_experiment(n_distinct, stream_length, epsilon, delta, num_trials=100):
    """Run multiple trials and analyze results"""
    print(f"\n=== EXPERIMENT ===")
    print(f"True F₀ = {n_distinct}")
    print(f"Stream length = {stream_length}")
    print(f"ε = {epsilon}, δ = {delta}")
    print(f"Theoretical threshold = {math.ceil(12 / (epsilon**2) * math.log(8 * stream_length / delta))}")
    print(f"Target range: [{(1-epsilon)*n_distinct:.1f}, {(1+epsilon)*n_distinct:.1f}]")
    
    estimates = []
    failures = 0
    final_ps = []
    final_Xs = []
    
    for trial in range(num_trials):
        # Generate a new stream for each trial
        stream = generate_stream(n_distinct, stream_length)
        
        # Verify the stream actually has n_distinct unique elements
        actual_distinct = len(set(stream))
        if actual_distinct != n_distinct:
            print(f"Warning: Stream has {actual_distinct} distinct elements, expected {n_distinct}")
        
        # Run the algorithm
        estimate, failed, final_p, final_X_size = f0_estimator(stream, epsilon, delta)
        
        if failed:
            failures += 1
        else:
            estimates.append(estimate)
            final_ps.append(final_p)
            final_Xs.append(final_X_size)
    
    # Analysis
    successful_trials = len(estimates)
    print(f"\n=== RESULTS ===")
    print(f"Successful trials: {successful_trials}/{num_trials}")
    print(f"Failure rate: {failures/num_trials:.3f} (theory predicts ≤ {delta/8:.3f})")
    
    if successful_trials > 0:
        # Accuracy analysis
        within_bounds = sum(1 for est in estimates 
                          if (1-epsilon)*n_distinct <= est <= (1+epsilon)*n_distinct)
        accuracy_rate = within_bounds / successful_trials
        
        print(f"\nAccuracy (among successful trials):")
        print(f"Within bounds: {within_bounds}/{successful_trials} = {accuracy_rate:.3f}")
        print(f"Theory predicts: ≥ {1-delta:.3f}")
        
        print(f"\nEstimate statistics:")
        print(f"Mean estimate: {sum(estimates)/len(estimates):.1f}")
        print(f"Min estimate: {min(estimates):.1f}")
        print(f"Max estimate: {max(estimates):.1f}")
        
        print(f"\nSampling statistics:")
        print(f"Final p values: min={min(final_ps):.6f}, max={max(final_ps):.6f}")
        print(f"Final |X| values: min={min(final_Xs)}, max={max(final_Xs)}")
        
        # Show some individual examples
        print(f"\nSample estimates: {estimates[:10]}")
        
        # Theory vs Practice comparison
        total_error_rate = (failures + (successful_trials - within_bounds)) / num_trials
        print(f"\n=== THEORY CHECK ===")
        print(f"Total error rate: {total_error_rate:.3f}")
        print(f"Theory allows: ≤ {delta:.3f}")
        print(f"✓ Theory validated!" if total_error_rate <= delta else "✗ Theory violated!")

# Run experiments with different parameters
random.seed(42)  # For reproducibility

# Experiment 1: Small example
print("EXPERIMENT 1: Small scale")
run_experiment(n_distinct=500, stream_length=1000, epsilon=0.1, delta=0.1, num_trials=200)

# Experiment 2: Medium scale  
print("\n" + "="*60)
print("EXPERIMENT 2: Medium scale")
#run_experiment(n_distinct=500, stream_length=5000, epsilon=0.2, delta=0.05, num_trials=100)

# Experiment 3: Large scale with tight bounds
print("\n" + "="*60)
print("EXPERIMENT 3: Large scale, tight bounds")
#run_experiment(n_distinct=1000, stream_length=10000, epsilon=0.05, delta=0.01, num_trials=50)

# Experiment 4: Test failure bounds specifically
print("\n" + "="*60)
print("EXPERIMENT 4: Testing failure rate bounds")
#run_experiment(n_distinct=50, stream_length=500, epsilon=0.3, delta=0.2, num_trials=500)

