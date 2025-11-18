#!/usr/bin/env node

// Embed C++ source code into HTML template
// Usage: node embed-code.js <cpp-file> <html-file>

const fs = require('fs');
const path = require('path');

if (process.argv.length < 4) {
	console.error('Usage: node embed-code.js <cpp-file> <html-file>');
	process.exit(1);
}

const cppFile = process.argv[2];
const htmlFile = process.argv[3];

try {
	// Check if files exist and are accessible
	if (!fs.existsSync(cppFile)) {
		throw new Error(`C++ file not found: ${cppFile}`);
	}
	if (!fs.existsSync(htmlFile)) {
		throw new Error(`HTML file not found: ${htmlFile}`);
	}

	// Check permissions
	try {
		fs.accessSync(cppFile, fs.constants.R_OK);
		fs.accessSync(htmlFile, fs.constants.R_OK | fs.constants.W_OK);
	} catch (err) {
		throw new Error(`Permission denied: ${err.message}`);
	}

	// Read C++ source code
	let cppContent = fs.readFileSync(cppFile, 'utf8');

	// escape the content
	cppContent = cppContent
		.replace(/"/g, '\\"')
		.replace(/\r/g, "")
		.replace(/\n/g, '\\n');

	// Read HTML file
	let htmlContent = fs.readFileSync(htmlFile, 'utf8');

	// Replace {{CODE}} with the escaped C++ content
	htmlContent = htmlContent.replace('{{CODE}}', cppContent);

	// Write back to HTML file
	fs.writeFileSync(htmlFile, htmlContent, 'utf8');

} catch (error) {
	console.error('[embed-code] Error:', error.message);
	console.error('[embed-code] Stack:', error.stack);
	process.exit(1);
}
