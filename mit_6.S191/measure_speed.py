#measure speed


import time

def speed_function(func):
	start_time = time.time()

	# Call your function
	func()

	# Calculate the elapsed time
	end_time = time.time()
	execution_time = end_time - start_time

	# Display the execution time
	print(f"Execution time: {execution_time} seconds")
	return 0

