import "./globals.css";

export const metadata = {
  title: "TailorResume Demo",
  description: "Deterministic resume-to-job matching demo",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
