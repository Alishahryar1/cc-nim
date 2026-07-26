'use client';

import { motion } from 'framer-motion';

export default function AboutPreview() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.2,
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
      whileInView="visible"
      viewport={{ once: true }}
      className="py-20 bg-dark"
    >
      <div className="container-custom">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Content */}
          <motion.div variants={itemVariants}>
            <h2 className="text-5xl font-bold mb-6">
              <span className="gradient-text">About</span> the philosophy
            </h2>
            <p className="text-lg text-white/70 mb-6 leading-relaxed">
              I believe great marketing begins with a great story. My work brings together
              strategy, content, creators, and campaigns to help brands create experiences
              people remember.
            </p>
            <p className="text-base text-white/60 leading-relaxed">
              Every project is approached with intentionality. I don't just create content
              or run campaigns  I craft narratives that resonate, engage, and convert.
              From strategy to execution, I'm invested in results that matter.
            </p>
          </motion.div>

          {/* Visual Element */}
          <motion.div
            variants={itemVariants}
            className="relative h-96 rounded-2xl overflow-hidden glass-effect"
          >
            <motion.div
              animate={{
                scale: [1, 1.05, 1],
                rotate: [0, 1, 0],
              }}
              transition={{
                duration: 6,
                repeat: Infinity,
              }}
              className="w-full h-full bg-gradient-to-br from-accent/20 to-accent-light/10 flex items-center justify-center"
            >
              <div className="text-center">
                <div className="text-6xl mb-4">📖</div>
                <p className="text-white/60">Every brand has a story</p>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </motion.section>
  );
}
