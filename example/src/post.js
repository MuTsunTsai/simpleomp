Module.onExit = function() {
	postMessage({ target: "custom" });
};