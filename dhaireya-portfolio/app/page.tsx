'use client';

import { motion } from 'framer-motion';
import HeroSection from '@/components/HeroSection';
import LogoShowcase from '@/components/LogoShowcase';
import AboutPreview from '@/components/AboutPreview';
import WorkShowcase from '@/components/WorkShowcase';
import ExperiencePreview from '@/components/ExperiencePreview';
import CTASection from '@/components/CTASection';

export default function Home() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.3,
      },
    },
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="min-h-screen bg-dark"
    >
      {/* Hero Section */}
      <HeroSection />

      {/* Logo Showcase Slideshow */}
      <LogoShowcase />

      {/* About Preview */}
      <AboutPreview />

      {/* Featured Work */}
      <WorkShowcase />

      {/* Experience Preview */}
      <ExperiencePreview />

      {/* CTA Section */}
      <CTASection />
    </motion.div>
  );
}
