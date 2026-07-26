'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { FiMenu, FiX } from 'react-icons/fi';

export default function Navigation() {
  const [isOpen, setIsOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navItems = [
    { label: 'Home', href: '/' },
    { label: 'About', href: '/about' },
    { label: 'Work', href: '/work' },
    { label: 'Experience', href: '/experience' },
    { label: 'Community', href: '/community' },
    { label: 'Contact', href: '/contact' },
  ];

  const navVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5 },
    },
  };

  const menuVariants = {
    hidden: { opacity: 0, x: 50 },
    visible: {
      opacity: 1,
      x: 0,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      },
    },
  };

  return (
    <motion.nav
      variants={navVariants}
      initial="hidden"
      animate="visible"
      className={`fixed w-full top-0 z-50 transition-all duration-300 ${
        isScrolled
          ? 'bg-dark-secondary/95 backdrop-blur-md border-b border-dark-tertiary'
          : 'bg-transparent'
      }`}
    >
      <div className="container-custom flex justify-between items-center py-6">
        {/* Logo */}
        <Link href="/" className="text-2xl font-bold gradient-text">
          DJ
        </Link>

        {/* Desktop Menu */}
        <div className="hidden md:flex gap-8">
          {navItems.map((item, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
            >
              <Link
                href={item.href}
                className="text-sm font-medium text-white/70 hover:text-accent transition-colors duration-300"
              >
                {item.label}
              </Link>
            </motion.div>
          ))}
        </div>

        {/* Mobile Menu Button */}
        <button
          className="md:hidden text-white text-xl"
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? <FiX /> : <FiMenu />}
        </button>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <motion.div
          variants={menuVariants}
          initial="hidden"
          animate="visible"
          className="md:hidden bg-dark-secondary/95 backdrop-blur-md border-b border-dark-tertiary"
        >
          <div className="container-custom py-6 flex flex-col gap-4">
            {navItems.map((item, i) => (
              <motion.div key={i} variants={menuVariants}>
                <Link
                  href={item.href}
                  className="block text-white/70 hover:text-accent transition-colors duration-300"
                  onClick={() => setIsOpen(false)}
                >
                  {item.label}
                </Link>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </motion.nav>
  );
}
