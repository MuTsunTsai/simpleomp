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
	console.error('Error:', error.message);
	process.exit(1);
}
