'use client';

import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';

export default function LogoShowcase() {
  const [currentLogo, setCurrentLogo] = useState(0);

  const logos = [
    { name: 'Arviend Sud', initials: 'AS', color: 'from-accent to-accent-light' },
    { name: 'Brand Strategy', initials: 'BS', color: 'from-blue-500 to-cyan-500' },
    { name: 'Content Creation', initials: 'CC', color: 'from-purple-500 to-pink-500' },
    { name: 'Marketing', initials: 'MK', color: 'from-green-500 to-emerald-500' },
    { name: 'Advertising', initials: 'AD', color: 'from-yellow-500 to-orange-500' },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentLogo((prev) => (prev + 1) % logos.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <motion.section
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      className="py-20 bg-dark-secondary relative overflow-hidden"
    >
      <div className="container-custom text-center">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-4xl font-bold mb-16 text-white"
        >
          Featured Brands & <span className="gradient-text">Collaborations</span>
        </motion.h2>

        {/* Logo Carousel */}
        <div className="flex justify-center items-center mb-8">
          <motion.div
            key={currentLogo}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.5 }}
            className={`w-40 h-40 rounded-2xl bg-gradient-to-br ${logos[currentLogo].color} flex items-center justify-center`}
          >
            <div className="text-center">
              <div className="text-5xl font-bold text-white mb-2">{logos[currentLogo].initials}</div>
              <div className="text-white text-sm font-semibold">{logos[currentLogo].name}</div>
            </div>
          </motion.div>
        </div>

        {/* Logo indicators */}
        <div className="flex justify-center gap-3">
          {logos.map((logo, i) => (
            <motion.button
              key={i}
              onClick={() => setCurrentLogo(i)}
              whileHover={{ scale: 1.2 }}
              className={`h-3 rounded-full transition-all duration-300 ${
                i === currentLogo ? 'w-8 bg-accent' : 'w-3 bg-white/20'
              }`}
            />
          ))}
        </div>

        {/* Logo Grid (below) */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
          className="mt-16 grid grid-cols-2 md:grid-cols-5 gap-4"
        >
          {logos.map((logo, i) => (
            <motion.div
              key={i}
              whileHover={{ y: -5 }}
              onClick={() => setCurrentLogo(i)}
              className={`h-24 rounded-lg flex items-center justify-center cursor-pointer transition-all duration-300 ${
                i === currentLogo
                  ? `bg-gradient-to-br ${logo.color}`
                  : 'bg-dark-tertiary hover:bg-dark-secondary'
              }`}
            >
              <div className="text-center">
                <div className={`text-2xl font-bold ${i === currentLogo ? 'text-white' : 'text-white/50'}`}>
                  {logo.initials}
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </motion.section>
  );
}
