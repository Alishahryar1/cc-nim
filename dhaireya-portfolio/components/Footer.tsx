'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { FiMail, FiLinkedin, FiInstagram, FiPhone } from 'react-icons/fi';

export default function Footer() {
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
    visible: { opacity: 1, y: 0 },
  };

  const socialLinks = [
    { icon: FiMail, href: 'mailto:jagya18dhaireya18@gmail.com', label: 'Email' },
    { icon: FiPhone, href: 'tel:+918448922579', label: 'Phone' },
    { icon: FiLinkedin, href: 'https://www.linkedin.com/in/dhaireya-jagya-298498272', label: 'LinkedIn' },
    { icon: FiInstagram, href: 'https://www.instagram.com/arviendsud', label: 'Instagram' },
  ];

  return (
    <motion.footer
      variants={containerVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      className="bg-dark-secondary border-t border-dark-tertiary py-16"
    >
      <div className="container-custom">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-12">
          {/* Brand */}
          <motion.div variants={itemVariants}>
            <h3 className="text-2xl font-bold gradient-text mb-4">Dhaireya Jagya</h3>
            <p className="text-white/50 text-sm">Building brands through stories people remember.</p>
          </motion.div>

          {/* Quick Links */}
          <motion.div variants={itemVariants}>
            <h4 className="text-white font-semibold mb-4">Pages</h4>
            <ul className="space-y-2 text-sm">
              {['About', 'Work', 'Experience', 'Community', 'Contact'].map((link) => (
                <li key={link}>
                  <Link href={`/${link.toLowerCase()}`} className="text-white/50 hover:text-accent transition-colors">
                    {link}
                  </Link>
                </li>
              ))}
            </ul>
          </motion.div>

          {/* Services */}
          <motion.div variants={itemVariants}>
            <h4 className="text-white font-semibold mb-4">Expertise</h4>
            <ul className="space-y-2 text-sm text-white/50">
              <li>Marketing Strategy</li>
              <li>Content Creation</li>
              <li>Advertising</li>
              <li>Campaign Management</li>
            </ul>
          </motion.div>

          {/* Location */}
          <motion.div variants={itemVariants}>
            <h4 className="text-white font-semibold mb-4">Location</h4>
            <p className="text-white/50 text-sm">Delhi, India</p>
            <p className="text-white/50 text-sm">Open to work anywhere</p>
          </motion.div>
        </div>

        {/* Social Links */}
        <motion.div
          variants={itemVariants}
          className="flex justify-center gap-6 py-8 border-t border-dark-tertiary"
        >
          {socialLinks.map((social, i) => {
            const Icon = social.icon;
            return (
              <motion.a
                key={i}
                href={social.href}
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.2 }}
                className="text-white/50 hover:text-accent transition-colors"
                title={social.label}
              >
                <Icon size={24} />
              </motion.a>
            );
          })}
        </motion.div>

        {/* Copyright */}
        <motion.div variants={itemVariants} className="text-center pt-8 text-white/30 text-sm">
          <p>© {new Date().getFullYear()} Dhaireya Jagya. All rights reserved.</p>
        </motion.div>
      </div>
    </motion.footer>
  );
}
