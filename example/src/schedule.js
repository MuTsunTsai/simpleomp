// We use --proxy-to-worker in the examples,
// so we cannot write preRun directly in template.html,
// and we need to write it here.

Module.preRun = [
	function() {
		// Setup environment variables
		ENV.OMP_SCHEDULE = "dynamic,1";
	}
];