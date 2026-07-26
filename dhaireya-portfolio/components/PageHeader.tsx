'use client';

import { motion } from 'framer-motion';

interface PageHeaderProps {
  title: string;
  subtitle: string;
}

export default function PageHeader({ title, subtitle }: PageHeaderProps) {
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

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.8 },
    },
  };

  return (
    <motion.section
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="min-h-[60vh] flex items-center justify-center bg-gradient-to-b from-dark-secondary to-dark pt-20"
    >
      <div className="container-custom text-center">
        <motion.h1
          variants={itemVariants}
          className="text-6xl md:text-7xl font-bold mb-6 gradient-text"
        >
          {title}
        </motion.h1>
        <motion.p
          variants={itemVariants}
          className="text-xl md:text-2xl text-white/70 max-w-2xl mx-auto"
        >
          {subtitle}
        </motion.p>
      </div>
    </motion.section>
  );
}
