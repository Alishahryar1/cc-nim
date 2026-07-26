'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { FiArrowRight } from 'react-icons/fi';

export default function ExperiencePreview() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: {
      opacity: 1,
      x: 0,
      transition: { duration: 0.8 },
    },
  };

  const experiences = [
    {
      title: 'Arviend Sud',
      role: 'Content Strategist & Scriptwriter',
      description: '60+ scripts created, 20+ email campaigns, 10+ WhatsApp campaigns with 10K-150K+ reach',
      impact: 'Content reaching millions with strategic storytelling',
    },
    {
      title: 'Brand Building',
      role: 'Marketing Strategist',
      description: 'Managed creator partnerships, built campaigns from strategy to execution',
      impact: 'End-to-end campaign management and coordination',
    },
    {
      title: 'Community Leadership',
      role: 'Secretary & Organizer',
      description: 'Revived inactive chapter, built team of 80+, organized events and collaborations',
      impact: 'Created community impact through leadership and coordination',
    },
  ];

  return (
    <motion.section
      variants={containerVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      className="py-20 bg-dark"
    >
      <div className="container-custom">
        <motion.div variants={itemVariants} className="text-center mb-16">
          <h2 className="text-5xl font-bold mb-4">
            Experience & <span className="gradient-text">Impact</span>
          </h2>
          <p className="text-white/60 text-lg max-w-2xl mx-auto">
            Strategic thinking combined with execution across marketing, content, and community
          </p>
        </motion.div>

        {/* Experience Timeline */}
        <div className="space-y-8 max-w-3xl mx-auto">
          {experiences.map((exp, i) => (
            <motion.div
              key={i}
              variants={itemVariants}
              className="relative pl-8 pb-8 border-l-2 border-accent/30 hover:border-accent transition-colors duration-300"
            >
              {/* Timeline dot */}
              <motion.div
                whileHover={{ scale: 1.5 }}
                className="absolute left-[-13px] top-0 w-6 h-6 rounded-full bg-accent"
              />

              <div className="glass-effect p-6 rounded-lg hover:border-accent/50 transition-all duration-300">
                <h3 className="text-2xl font-bold text-accent mb-2">{exp.title}</h3>
                <p className="text-white font-semibold mb-3">{exp.role}</p>
                <p className="text-white/70 mb-3">{exp.description}</p>
                <p className="text-sm text-white/50 italic">{exp.impact}</p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* CTA */}
        <motion.div variants={itemVariants} className="text-center mt-16">
          <Link
            href="/experience"
            className="inline-flex items-center gap-3 px-8 py-4 bg-accent text-dark font-semibold rounded-lg hover:bg-accent-light transition-all duration-300 group"
          >
            View Full Experience
            <FiArrowRight className="group-hover:translate-x-2 transition-transform" />
          </Link>
        </motion.div>
      </div>
    </motion.section>
  );
}
