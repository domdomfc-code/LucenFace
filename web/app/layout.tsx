import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin", "vietnamese"],
  variable: "--font-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "LucenFace — Chuẩn hóa ảnh chân dung",
  description:
    "Kiểm tra & xử lý ảnh thẻ: crop, nền xanh, batch tới 50 ảnh.",
  icons: {
    icon: "/lucenface-logo.png",
    apple: "/lucenface-logo.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi" className={inter.variable}>
      <body className="font-sans">{children}</body>
    </html>
  );
}
