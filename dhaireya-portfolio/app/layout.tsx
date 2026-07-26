import type { Metadata } from 'next';
import './globals.css';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';

export const metadata: Metadata = {
  title: 'Dhaireya Jagya | Marketing & Storytelling',
  description: 'Building Brands Through Stories People Remember. Marketing strategist, content creator, and campaign manager.',
  keywords: 'marketing, storytelling, content strategy, social media, advertising, brand marketing',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-dark">
        <Navigation />
        <main>{children}</main>
        <Footer />
      </body>
    </html>
  );
}
