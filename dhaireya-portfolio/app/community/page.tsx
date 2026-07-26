'use client';

import { motion } from 'framer-motion';
import PageHeader from '@/components/PageHeader';
import CommunityContent from '@/components/sections/CommunityContent';

export default function Community() {
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
        title="Community & Impact"
        subtitle="Leading, inspiring, and creating positive change"
      />
      <CommunityContent />
    </motion.div>
  );
}
