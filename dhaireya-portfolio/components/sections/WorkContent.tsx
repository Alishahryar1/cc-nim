'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';
import Link from 'next/link';

export default function WorkContent() {
  const [selectedWork, setSelectedWork] = useState(0);

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
      description: 'A deep-dive numerology analysis exploring the connection between birth dates and prosperity.',
      topics: ['Numerology', 'Cultural Insights', 'Predictions'],
      link: 'https://youtu.be/F0HkD1H81bk?si=K6ERFF-gOBQwbV9-',
      reach: '50K+',
    },
    {
      type: 'YouTube Analysis',
      title: 'The Real Reason Behind Nitin Gadkari\'s Fall',
      category: 'Political Analysis',
      description: 'Numerology-based analysis of political events and frequency clashes affecting public figures.',
      topics: ['Political Analysis', 'Numerology', '2026-2027 Predictions'],
      link: 'https://youtu.be/8I7OaUcCJpk?si=BYWprZ5_DgZ7ytP2',
      reach: '75K+',
    },
    {
      type: 'Monthly Predictions',
      title: 'What August 2026 Has in Store for Your Driver Number',
      category: 'Personal Predictions',
      description: 'Comprehensive August 2026 predictions for all driver numbers with financial and relationship insights.',
      topics: ['Driver Numbers', 'Monthly Forecast', 'Karmic Analysis'],
      link: 'https://youtu.be/5f_QC_Z08iI?si=j0uOpYQ4fDZWeKvK',
      reach: '100K+',
    },
    {
      type: 'Instagram Reel',
      title: 'Will Samay Raina Face Another Controversy?',
      category: 'Celebrity Analysis',
      description: 'Short-form video analyzing celebrity numerology and predicting potential challenges.',
      topics: ['Celebrity', 'Numerology', 'Predictions'],
      link: 'https://www.instagram.com/reel/Da5aR_qJFAP/',
      reach: '150K+',
    },
  ];

  return (
    <motion.section
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="py-20 bg-dark"
    >
      <div className="container-custom">
        {/* Work Showcase */}
        <div className="max-w-4xl mx-auto">
          {/* Featured Work */}
          <motion.div
            key={selectedWork}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-12 p-8 rounded-2xl glass-effect border border-accent/30"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <span className="text-accent text-sm font-semibold">{works[selectedWork].type}</span>
                <h2 className="text-3xl font-bold text-white mt-3 mb-4 leading-tight">
                  {works[selectedWork].title}
                </h2>
                <p className="text-white/70 mb-6 leading-relaxed">
                  {works[selectedWork].description}
                </p>

                {/* Topics */}
                <div className="flex flex-wrap gap-2 mb-6">
                  {works[selectedWork].topics.map((topic, i) => (
                    <span
                      key={i}
                      className="px-3 py-1 rounded-full bg-accent/20 text-accent text-xs font-semibold"
                    >
                      {topic}
                    </span>
                  ))}
                </div>

                {/* Reach */}
                <p className="text-white/50 mb-6 text-sm">
                  Reach: <span className="text-accent font-bold">{works[selectedWork].reach}</span>
                </p>

                {/* CTA */}
                <a
                  href={works[selectedWork].link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-block px-6 py-3 bg-accent text-dark font-semibold rounded-lg hover:bg-accent-light transition-all duration-300"
                >
                  View on Platform →
                </a>
              </div>

              {/* Visual */}
              <motion.div
                animate={{
                  scale: [1, 1.02, 1],
                }}
                transition={{
                  duration: 4,
                  repeat: Infinity,
                }}
                className="h-80 rounded-xl bg-gradient-to-br from-accent/20 to-accent-light/10 flex items-center justify-center"
              >
                <div className="text-center">
                  <div className="text-6xl mb-4">▶</div>
                  <p className="text-white/60">{works[selectedWork].type}</p>
                </div>
              </motion.div>
            </div>
          </motion.div>

          {/* Work Navigation */}
          <div className="space-y-3">
            {works.map((work, i) => (
              <motion.button
                key={i}
                onClick={() => setSelectedWork(i)}
                whileHover={{ x: 10 }}
                className={`w-full text-left p-6 rounded-xl transition-all duration-300 ${
                  i === selectedWork
                    ? 'bg-accent/20 border-2 border-accent'
                    : 'bg-dark-tertiary border-2 border-transparent hover:bg-dark-secondary'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <p className="text-accent text-xs font-semibold uppercase">{work.type}</p>
                    <h4 className="text-white font-bold mt-1 text-lg line-clamp-2">{work.title}</h4>
                  </div>
                  <span className="text-white/50 text-sm ml-4">{work.reach}</span>
                </div>
              </motion.button>
            ))}
          </div>
        </div>
      </div>
    </motion.section>
  );
}
