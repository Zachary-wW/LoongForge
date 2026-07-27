mkdir -p /workspace/bridge_test_log/minicpm_v_4_6/

sh tools/dist_checkpoint/test/minicpm_v_4_6/minicpm_v_4_6_bridge_roundtrip.sh \
    2>&1 | tee -a /workspace/bridge_test_log/minicpm_v_4_6/minicpm_v_4_6_log
