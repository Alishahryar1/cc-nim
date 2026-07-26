'use client';

import { motion } from 'framer-motion';
import PageHeader from '@/components/PageHeader';
import AboutContent from '@/components/sections/AboutContent';

export default function About() {
  const pageVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 },
    },
  };

  return (
    <motion.div
      variants={pageVariants}
      initial="hidden"
      animate="visible"
      className="min-h-screen bg-dark"
    >
      <PageHeader
        title="About"
        subtitle="Understanding the philosophy behind the work"
      />
      <AboutContent />
    </motion.div>
  );
}
