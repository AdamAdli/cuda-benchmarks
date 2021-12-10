FUNCTION(detect_cluster)
    EXECUTE_PROCESS(
        COMMAND bash -c "squeue --help" OUTPUT_QUIET
        RESULT_VARIABLE CHECK_SLURM_TEST_RESULT
    )

    IF(NOT ${CHECK_SLURM_TEST_RESULT} EQUAL 0)
        # No slurm found, we are likely not on a cluster
        SET(CLUSTER COMPUTE_CANADA_GRAHAM PARENT_SCOPE)
        RETURN()
    ENDIF()

    EXECUTE_PROCESS(
        COMMAND bash -c "squeue --nodelist=gra100" OUTPUT_QUIET
        RESULT_VARIABLE NODE_CHECK_RESULT
    )

    IF(${NODE_CHECK_RESULT} EQUAL 0)
        MESSAGE(STATUS "gra100 node found, assuming we are on the Graham cluster")
        SET(CLUSTER COMPUTE_CANADA_GRAHAM PARENT_SCOPE)
        RETURN()
    ENDIF()

    IF(${NODE_CHECK_RESULT} EQUAL 0)
        MESSAGE(STATUS "nia100 node found, assuming we are on the Graham cluster")
        SET(CLUSTER COMPUTE_CANADA_NIAGARA PARENT_SCOPE)
        RETURN()
    ENDIF()

    # We detected slurm but not a known node, so we are on a cluster but can't identify it
    #   so return UNKNOWN_CLUSTER
    SET(CLUSTER UNKNOWN_CLUSTER PARENT_SCOPE)
ENDFUNCTION()