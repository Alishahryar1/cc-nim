'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { FiMail, FiPhone, FiLinkedin } from 'react-icons/fi';

export default function CTASection() {
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
      className="py-24 bg-gradient-to-b from-dark to-dark-secondary relative overflow-hidden"
    >
      {/* Background decoration */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          animate={{
            scale: [1, 1.1, 1],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
          }}
          className="absolute -top-40 right-0 w-96 h-96 bg-accent/5 rounded-full blur-3xl"
        />
      </div>

      <div className="container-custom relative z-10">
        <motion.div variants={itemVariants} className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-5xl md:text-6xl font-bold mb-6">
            Ready to collaborate?
          </h2>
          <p className="text-xl text-white/70">
            Whether you need a marketing strategist, content creator, or campaign manager,
            let's create something memorable together.
          </p>
        </motion.div>

        {/* Contact Options */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {[
            {
              icon: FiMail,
              label: 'Email',
              value: 'jagya18dhaireya18@gmail.com',
              href: 'mailto:jagya18dhaireya18@gmail.com',
            },
            {
              icon: FiPhone,
              label: 'Phone',
              value: '+91 8448922579',
              href: 'tel:+918448922579',
            },
            {
              icon: FiLinkedin,
              label: 'LinkedIn',
              value: 'Dhaireya Jagya',
              href: 'https://www.linkedin.com/in/dhaireya-jagya-298498272',
            },
          ].map((contact, i) => {
            const Icon = contact.icon;
            return (
              <motion.a
                key={i}
                variants={itemVariants}
                href={contact.href}
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ y: -5 }}
                className="p-6 rounded-xl glass-effect hover:border-accent/50 transition-all duration-300 cursor-pointer"
              >
                <Icon className="text-accent text-3xl mb-3" />
                <p className="text-white/50 text-sm font-semibold mb-1">{contact.label}</p>
                <p className="text-white font-semibold line-clamp-2">{contact.value}</p>
              </motion.a>
            );
          })}
        </div>

        {/* Primary CTA Button */}
        <motion.div variants={itemVariants} className="text-center">
          <Link
            href="/contact"
            className="inline-flex items-center gap-3 px-10 py-5 bg-accent text-dark font-bold text-lg rounded-lg hover:bg-accent-light transition-all duration-300 group shadow-lg hover:shadow-xl"
          >
            Let's Talk
            <span className="text-2xl">→</span>
          </Link>
        </motion.div>

        {/* Location Info */}
        <motion.div variants={itemVariants} className="text-center mt-12 text-white/50">
          <p>Delhi, India  Available for opportunities worldwide</p>
        </motion.div>
      </div>
    </motion.section>
  );
}
