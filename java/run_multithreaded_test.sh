#!/bin/bash

# Script to run CagraMultiThreadStabilityIT test 100 times and collect results

cd /home/ishan/code/cuvs/java/cuvs-java
export LD_LIBRARY_PATH=/home/ishan/code/cuvs/cpp/build

# Initialize counters
TOTAL_RUNS=10
PASSED=0
FAILED=0
ERRORS=0

# Create results file
RESULTS_FILE="multithreaded_test_results.txt"
echo "Running CagraMultiThreadStabilityIT $TOTAL_RUNS times..." > $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE

# Run the test multiple times
for i in $(seq 1 $TOTAL_RUNS); do
    echo -n "Run $i/$TOTAL_RUNS: "
    
    # Run the test and capture output with 30 second timeout
    OUTPUT=$(timeout 30s mvn integration-test -Dit.test=com.nvidia.cuvs.CagraMultiThreadStabilityIT -q 2>&1)
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "PASSED"
        echo "Run $i: PASSED" >> $RESULTS_FILE
        ((PASSED++))
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "TIMEOUT"
        echo "Run $i: TIMEOUT (30 seconds)" >> $RESULTS_FILE
        ((FAILED++))
        ((ERRORS++))
    else
        echo "FAILED"
        echo "Run $i: FAILED (Exit code: $EXIT_CODE)" >> $RESULTS_FILE
        echo "Error output:" >> $RESULTS_FILE
        echo "$OUTPUT" | grep -E "(ERROR|FAILURE|Exception|SIGSEGV)" >> $RESULTS_FILE 2>/dev/null || echo "No specific error found" >> $RESULTS_FILE
        echo "---" >> $RESULTS_FILE
        ((FAILED++))
        
        # Check if it's a crash/error vs test failure
        if echo "$OUTPUT" | grep -q "SIGSEGV\|core dumped\|JVM crash"; then
            ((ERRORS++))
        fi
    fi
done

# Summary
echo "========================================" >> $RESULTS_FILE
echo "SUMMARY:" >> $RESULTS_FILE
echo "Total runs: $TOTAL_RUNS" >> $RESULTS_FILE
echo "Passed: $PASSED" >> $RESULTS_FILE
echo "Failed: $FAILED" >> $RESULTS_FILE
echo "Crashes/Errors: $ERRORS" >> $RESULTS_FILE
echo "Success rate: $(( PASSED * 100 / TOTAL_RUNS ))%" >> $RESULTS_FILE

# Print summary to console
echo ""
echo "Test execution completed!"
echo "========================="
echo "Total runs: $TOTAL_RUNS"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "Crashes/Errors: $ERRORS"
echo "Success rate: $(( PASSED * 100 / TOTAL_RUNS ))%"
echo ""
echo "Detailed results saved to: $RESULTS_FILE"