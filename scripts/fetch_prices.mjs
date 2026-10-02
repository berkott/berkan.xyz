// Fetch 1y of daily closes from Yahoo Finance and write static/financial_data/prices.json.
//
// Runs on the host (not the browser), so there is no CORS involved and no proxy
// needed. Wired to `predev` in package.json, so `npm run dev` refreshes prices.
//
// The finances page and its data are gitignored, so this is a no-op on a clean
// checkout (CI, fresh clone) — it exits 0 without writing anything.

import { readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const CSV_PATH = resolve(ROOT, 'static/financial_data/master_investments.csv');
const OUT_PATH = resolve(ROOT, 'static/financial_data/prices.json');

const YAHOO = (t) =>
	`https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(t)}?interval=1d&range=1y`;

// Rows whose value is a fixed number (cash, loans) carry no ticker to price.
const STATIC_KEYWORDS = ['cash', 'loan', 'loans'];

function tradeableTickers(csv) {
	const lines = csv.trim().split('\n');
	const tickers = new Set();

	for (let i = 1; i < lines.length; i++) {
		const [ticker, , currentValue] = lines[i].split(',');
		if (!ticker?.trim()) continue;
		if (currentValue?.trim()) continue;
		if (STATIC_KEYWORDS.some((kw) => ticker.toLowerCase().includes(kw))) continue;
		tickers.add(ticker.trim());
	}

	return [...tickers];
}

async function fetchDailyCloses(ticker) {
	const res = await fetch(YAHOO(ticker), { headers: { 'User-Agent': 'Mozilla/5.0' } });
	if (!res.ok) throw new Error(`HTTP ${res.status}`);

	const result = (await res.json())?.chart?.result?.[0];
	const timestamps = result?.timestamp;
	const closes = result?.indicators?.quote?.[0]?.close;
	if (!timestamps || !closes) throw new Error('unexpected response shape');

	// Key by UTC date, matching how the page previously bucketed these.
	const byDay = {};
	for (let i = 0; i < timestamps.length; i++) {
		if (closes[i] == null) continue;
		byDay[new Date(timestamps[i] * 1000).toISOString().slice(0, 10)] = closes[i];
	}

	return byDay;
}

async function main() {
	if (!existsSync(CSV_PATH)) {
		console.log('[prices] no master_investments.csv — skipping');
		return;
	}

	const tickers = tradeableTickers(await readFile(CSV_PATH, 'utf8'));
	console.log(`[prices] fetching ${tickers.length} tickers...`);

	const entries = await Promise.all(
		tickers.map(async (ticker) => {
			try {
				const byDay = await fetchDailyCloses(ticker);
				console.log(`[prices]   ${ticker}: ${Object.keys(byDay).length} days`);
				return [ticker, byDay];
			} catch (e) {
				console.error(`[prices]   ${ticker}: FAILED (${e.message})`);
				return [ticker, null];
			}
		})
	);

	const failed = entries.filter(([, v]) => v === null).map(([t]) => t);
	const ok = entries.filter(([, v]) => v !== null);

	// Never overwrite good data with nothing; a stale chart beats an empty one.
	if (ok.length === 0) {
		console.error('[prices] every ticker failed — keeping existing prices.json');
		return;
	}

	await writeFile(
		OUT_PATH,
		JSON.stringify(
			{ generated: new Date().toISOString(), range: '1y', tickers: Object.fromEntries(ok) },
			null,
			'\t'
		)
	);

	console.log(`[prices] wrote ${ok.length} tickers to prices.json`);
	if (failed.length) console.error(`[prices] missing: ${failed.join(', ')}`);
}

main().catch((e) => {
	// A price refresh must never block `npm run dev`.
	console.error(`[prices] ${e.message} — keeping existing prices.json`);
});
