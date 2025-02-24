import sympy as sp
from SumOfSquares import SOSProblem, poly_variable
from multiprocessing import Pool
import time
import os

# Function to execute the Sum-of-Squares (SOS) problem
# 100 times take 17.81s for polynomial of degree 2
R = 2
xt, yt, zt, t = sp.symbols('xt yt zt t')
s1 =poly_variable('c1', [t], 2)
s2 =poly_variable('c2', [t], 2)
p1 = s1 * t**2 - s2 * (R**2- xt**2 - yt**2 - zt**2)

def run_sos_task(_):
    prob = SOSProblem()
    const = prob.add_sos_constraint(s1, [t])
    const = prob.add_sos_constraint(s2, [t])
    const = prob.add_sos_constraint(p1, [xt, yt, zt, t])

    
    prob.solve(solver="mosek", verbosity=False)

    # Prints Sum-of-Squares decomposition
    return(sum(const.get_sos_decomp()))



# Main function to execute tasks in parallel
if __name__ == "__main__":
    # run_sos_task(1)
    # # Number of parallel tasks
    num_tasks = 90
    results = []

    # Start timing
    start_time = time.time()


    with Pool(os.cpu_count()) as pool:
        for result in pool.imap(run_sos_task, range(num_tasks)):
            results.append(result)
        print(len(results))

    # End timing
    end_time = time.time()

    # Calculate total time taken
    total_time = end_time - start_time
    print(f"Total time for {num_tasks} tasks: {total_time:.2f} seconds")

    # Optionally, print all results
    # for idx, result in enumerate(results[:5]):  # Print only the first 5 results for brevity
    #     print(f"Result of task {idx + 1}: {result}")

    pool.terminate()

