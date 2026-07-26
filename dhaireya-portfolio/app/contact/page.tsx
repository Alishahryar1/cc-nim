'use client';

import { motion } from 'framer-motion';
import PageHeader from '@/components/PageHeader';
import ContactContent from '@/components/sections/ContactContent';

export default function Contact() {
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
        title="Get In Touch"
        subtitle="Let's collaborate on something amazing"
      />
      <ContactContent />
    </motion.div>
  );
}
