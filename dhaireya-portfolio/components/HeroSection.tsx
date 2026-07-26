'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { FiArrowRight } from 'react-icons/fi';

export default function HeroSection() {
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
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.8, ease: 'easeOut' },
    },
  };

  return (
    <motion.section
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="relative min-h-screen flex items-center justify-center pt-20 bg-gradient-to-b from-dark-secondary via-dark to-dark overflow-hidden"
    >
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            rotate: [0, 180, 360],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: 'linear',
          }}
          className="absolute -top-40 -right-40 w-80 h-80 bg-accent/5 rounded-full blur-3xl"
        />
        <motion.div
          animate={{
            scale: [1.2, 1, 1.2],
            rotate: [360, 180, 0],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: 'linear',
          }}
          className="absolute -bottom-40 -left-40 w-80 h-80 bg-accent/5 rounded-full blur-3xl"
        />
      </div>

      <div className="container-custom relative z-10 text-center max-w-4xl">
        {/* Main Hero Statement */}
        <motion.div variants={itemVariants} className="mb-8">
          <h1 className="text-7xl md:text-8xl font-bold leading-tight mb-6">
            <span className="gradient-text block">Building Brands</span>
            <span className="text-white">Through Stories</span>
            <span className="gradient-text block">People Remember</span>
          </h1>
        </motion.div>

        {/* Supporting Line */}
        <motion.p
          variants={itemVariants}
          className="text-xl md:text-2xl text-white/70 mb-8 leading-relaxed"
        >
          I help brands connect with people through marketing, storytelling, creator partnerships,
          advertising, and campaigns that leave a lasting impression.
        </motion.p>

        {/* Name Introduction */}
        <motion.div variants={itemVariants} className="mb-12">
          <p className="text-lg text-white/50 mb-4">Hi, I'm Dhaireya Jagya</p>
          <div className="flex flex-wrap justify-center gap-3 text-sm">
            {[
              'Marketing Strategist',
              'Content Creator',
              'Campaign Manager',
              'Brand Builder',
            ].map((tag, i) => (
              <motion.span
                key={i}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.8 + i * 0.1 }}
                className="px-4 py-2 rounded-full glass-effect text-accent"
              >
                {tag}
              </motion.span>
            ))}
          </div>
        </motion.div>

        {/* CTA Buttons */}
        <motion.div
          variants={itemVariants}
          className="flex flex-col sm:flex-row gap-6 justify-center items-center"
        >
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Link
              href="/work"
              className="inline-flex items-center gap-3 px-8 py-4 bg-accent text-dark font-semibold rounded-lg hover:bg-accent-light transition-all duration-300 group"
            >
              Explore My Work
              <FiArrowRight className="group-hover:translate-x-2 transition-transform" />
            </Link>
          </motion.div>

          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Link
              href="/contact"
              className="inline-flex items-center gap-3 px-8 py-4 border-2 border-accent text-accent font-semibold rounded-lg hover:bg-accent/10 transition-all duration-300"
            >
              Get In Touch
            </Link>
          </motion.div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="absolute bottom-10 left-1/2 transform -translate-x-1/2 text-white/30"
        >
          <svg
            className="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 14l-7 7m0 0l-7-7m7 7V3"
            />
          </svg>
        </motion.div>
      </div>
    </motion.section>
  );
}
