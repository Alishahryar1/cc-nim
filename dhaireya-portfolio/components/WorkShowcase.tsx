'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { FiArrowRight } from 'react-icons/fi';

export default function WorkShowcase() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
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

  const works = [
    {
      type: 'YouTube Script',
      title: 'Why Women Born on These Dates Are Called Ghar Ki Lakshmi',
      category: 'Numerology & Culture',
      link: 'https://youtu.be/F0HkD1H81bk',
    },
    {
      type: 'Instagram Reel',
      title: 'Will Samay Raina Face Another Controversy?',
      category: 'Celebrity Analysis',
      link: 'https://www.instagram.com/reel/Da5aR_qJFAP/',
    },
    {
      type: 'YouTube Analysis',
      title: 'The Real Reason Behind Nitin Gadkari\'s Fall',
      category: 'Political Analysis',
      link: 'https://youtu.be/8I7OaUcCJpk',
    },
  ];

  return (
    <motion.section
      variants={containerVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      className="py-20 bg-dark-secondary"
    >
      <div className="container-custom">
        <motion.div variants={itemVariants} className="text-center mb-16">
          <h2 className="text-5xl font-bold mb-4">
            Featured <span className="gradient-text">Work</span>
          </h2>
          <p className="text-white/60 text-lg max-w-2xl mx-auto">
            Selected scripts, campaigns, and creative storytelling that brought ideas to life
          </p>
        </motion.div>

        {/* Work Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
          {works.map((work, i) => (
            <motion.div
              key={i}
              variants={itemVariants}
              whileHover={{ y: -10 }}
              className="group"
            >
              <a
                href={work.link}
                target="_blank"
                rel="noopener noreferrer"
                className="h-full block p-8 rounded-xl glass-effect hover:glass-effect transition-all duration-300 cursor-pointer border border-transparent hover:border-accent/50"
              >
                <div className="text-accent text-sm font-semibold mb-3 group-hover:scale-110 transition-transform">
                  {work.type}
                </div>
                <h3 className="text-xl font-bold text-white mb-3 group-hover:text-accent transition-colors line-clamp-2">
                  {work.title}
                </h3>
                <p className="text-white/50 text-sm mb-4">{work.category}</p>
                <div className="flex items-center gap-2 text-accent text-sm font-semibold group-hover:gap-3 transition-all">
                  View Details
                  <FiArrowRight className="group-hover:translate-x-1 transition-transform" />
                </div>
              </a>
            </motion.div>
          ))}
        </div>

        {/* View All Link */}
        <motion.div variants={itemVariants} className="text-center">
          <Link
            href="/work"
            className="inline-flex items-center gap-3 px-8 py-4 border-2 border-accent text-accent font-semibold rounded-lg hover:bg-accent/10 transition-all duration-300"
          >
            View All Work
            <FiArrowRight />
          </Link>
        </motion.div>
      </div>
    </motion.section>
  );
}
