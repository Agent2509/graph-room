import type { Metadata } from "next";
import { Fraunces, Inter_Tight } from "next/font/google";
import Link from "next/link";

import JobWatcher from "@/components/JobWatcher";
import StatusBar from "@/components/StatusBar";
import "./globals.css";

const display = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

const ui = Inter_Tight({
  subsets: ["latin"],
  variable: "--font-ui",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Graph Room",
  description: "Knowledge graph RAG: ingest documents, inspect the graph, ask questions.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${ui.variable}`}>
      <body>
        <div className="shell">
          <header className="topbar">
            <p className="wordmark">
              Graph<span>·</span>Room
            </p>
            <nav className="nav">
              <Link href="/">Workspace</Link>
              <Link href="/graph/">Graph</Link>
            </nav>
            <StatusBar />
          </header>
          <JobWatcher />
          {children}
        </div>
      </body>
    </html>
  );
}
