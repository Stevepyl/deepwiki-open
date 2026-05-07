import type { Metadata } from "next";
import "./globals.css";
import { LanguageProvider } from "../contexts/LanguageContext";
import { SettingsProvider } from "../contexts/SettingsContext";

export const metadata: Metadata = {
  title: "OpsWiki",
  description: "OpsWiki - Chat with code",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh">
      <body>
        <LanguageProvider>
          <SettingsProvider>{children}</SettingsProvider>
        </LanguageProvider>
      </body>
    </html>
  );
}
