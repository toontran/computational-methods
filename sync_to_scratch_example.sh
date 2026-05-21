nohup env OUTPUT_NAME=output TARGET_END_IDX=1138 STABILITY_SECONDS=60 CHECK_INTERVAL_SECONDS=60 ZIP_AFTER_SYNC=1 ZIP_SAMPLE_M=50 ZIP_OUTPUT_DIR=/scratch/ttran02/output_sampled_zips bash sync_to_scratch.sh > sync_to_scratch.nohup.out 2>&1 &
# tail -f sync_to_scratch.nohup.out
# cat .sync_to_scratch_output.state
# ps -fu "$USER" | grep -i sync_to_scratch | grep -v grep
