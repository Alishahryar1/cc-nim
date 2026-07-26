'use client';

import { motion } from 'framer-motion';
import PageHeader from '@/components/PageHeader';
import ExperienceContent from '@/components/sections/ExperienceContent';

export default function Experience() {
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
        title="Experience"
        subtitle="Building brands, managing teams, and executing campaigns"
      />
      <ExperienceContent />
    </motion.div>
  );
}
