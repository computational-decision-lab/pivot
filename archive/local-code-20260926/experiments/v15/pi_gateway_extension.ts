/**
 * Route the pinned Pi provider through the project-local compatible gateway.
 *
 * The extension changes only the base URL.  Pi still resolves
 * ANTHROPIC_AUTH_TOKEN and emits its normal bearer authentication header;
 * no credential is embedded in this file or any provenance artifact.
 */
export default function registerPivotGateway(pi: { registerProvider: (name: string, config: { baseUrl: string }) => void }): void {
	const baseUrl = process.env.ANTHROPIC_BASE_URL;
	if (!baseUrl) {
		throw new Error("ANTHROPIC_BASE_URL is required for the pinned Pi gateway extension");
	}
	pi.registerProvider("anthropic", { baseUrl });
}
