from kstream import KStream

stream = KStream()
stream.run_non_segmented(duration=10)
