/**
 * API client for backend communication
 */

// Use direct backend connection to bypass Vite proxy for large uploads
const API_BASE = 'http://localhost:8020/api/v1';

/**
 * Upload PCAP file
 * @param {File} file
 * @returns {Promise<Object>}
 */
export async function uploadPcap(file) {
	const formData = new FormData();
	formData.append('file', file);

	const response = await fetch(`${API_BASE}/upload/`, {
		method: 'POST',
		body: formData
	});

	if (!response.ok) {
		const error = await response.json();
		throw new Error(error.detail || 'Upload failed');
	}

	return response.json();
}

/**
 * Get job status
 * @param {string} jobId
 * @returns {Promise<Object>}
 */
export async function getJobStatus(jobId) {
	const response = await fetch(`${API_BASE}/upload/${jobId}`);

	if (!response.ok) {
		throw new Error('Failed to get job status');
	}

	return response.json();
}

/**
 * List all jobs
 * @param {number} limit
 * @param {number} offset
 * @returns {Promise<Array>}
 */
export async function listJobs(limit = 50, offset = 0) {
	const response = await fetch(`${API_BASE}/upload/?limit=${limit}&offset=${offset}`);

	if (!response.ok) {
		throw new Error('Failed to list jobs');
	}

	return response.json();
}

/**
 * Get packets for a job
 * @param {string} jobId
 * @param {Object} filters
 * @returns {Promise<Object>}
 */
export async function getPackets(jobId, filters = {}) {
	const params = new URLSearchParams({ job_id: jobId, ...filters });
	const response = await fetch(`${API_BASE}/packets?${params}`);

	if (!response.ok) {
		throw new Error('Failed to get packets');
	}

	return response.json();
}

/**
 * Get flows for a job
 * @param {string} jobId
 * @param {Object} filters
 * @returns {Promise<Object>}
 */
export async function getFlows(jobId, filters = {}) {
	const params = new URLSearchParams({ job_id: jobId, ...filters });
	const response = await fetch(`${API_BASE}/flows?${params}`);

	if (!response.ok) {
		throw new Error('Failed to get flows');
	}

	return response.json();
}

/**
 * Get job summary statistics
 * @param {string} jobId
 * @returns {Promise<Object>}
 */
export async function getJobSummary(jobId) {
	const response = await fetch(`${API_BASE}/stats/summary?job_id=${jobId}`);

	if (!response.ok) {
		throw new Error('Failed to get job summary');
	}

	return response.json();
}

/**
 * Get protocol distribution
 * @param {string} jobId
 * @returns {Promise<Object>}
 */
export async function getProtocolDistribution(jobId) {
	const response = await fetch(`${API_BASE}/stats/protocols?job_id=${jobId}`);

	if (!response.ok) {
		throw new Error('Failed to get protocol distribution');
	}

	return response.json();
}

/**
 * Get network topology graph data
 * @param {string} jobId
 * @param {number} limit
 * @returns {Promise<Object>}
 */
export async function getNetworkTopology(jobId, limit = 500) {
	const response = await fetch(`${API_BASE}/graph/topology?job_id=${jobId}&limit=${limit}`);

	if (!response.ok) {
		throw new Error('Failed to get network topology');
	}

	return response.json();
}

/**
 * Create EventSource for job status streaming
 * @param {string} jobId
 * @returns {EventSource}
 */
export function streamJobStatus(jobId) {
	return new EventSource(`${API_BASE}/stream/job/${jobId}`);
}
