'use client';

import { motion } from 'framer-motion';
import PageHeader from '@/components/PageHeader';
import WorkContent from '@/components/sections/WorkContent';

export default function Work() {
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
        title="Selected Work"
        subtitle="Scripts, campaigns, and creative storytelling"
      />
      <WorkContent />
    </motion.div>
  );
}
