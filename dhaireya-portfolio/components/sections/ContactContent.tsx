'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';
import { FiMail, FiPhone, FiLinkedin, FiInstagram, FiMapPin } from 'react-icons/fi';

export default function ContactContent() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: '',
  });
  const [submitted, setSubmitted] = useState(false);

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

  const handleChange = (e: any) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = (e: any) => {
    e.preventDefault();
    setSubmitted(true);
    setTimeout(() => setSubmitted(false), 3000);
    setFormData({ name: '', email: '', subject: '', message: '' });
  };

  const contactMethods = [
    {
      icon: FiMail,
      label: 'Email',
      value: 'jagya18dhaireya18@gmail.com',
      href: 'mailto:jagya18dhaireya18@gmail.com',
      description: 'Best for detailed inquiries',
    },
    {
      icon: FiPhone,
      label: 'Phone',
      value: '+91 8448922579',
      href: 'tel:+918448922579',
      description: 'Quick conversations',
    },
    {
      icon: FiLinkedin,
      label: 'LinkedIn',
      value: 'Dhaireya Jagya',
      href: 'https://www.linkedin.com/in/dhaireya-jagya-298498272',
      description: 'Professional networking',
    },
    {
      icon: FiInstagram,
      label: 'Instagram',
      value: '@arviendsud',
      href: 'https://www.instagram.com/arviendsud',
      description: 'Latest updates',
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
        {/* Contact Methods */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-20">
          {contactMethods.map((method, i) => {
            const Icon = method.icon;
            return (
              <motion.a
                key={i}
                variants={itemVariants}
                href={method.href}
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ y: -5 }}
                className="p-6 rounded-xl glass-effect hover:border-accent/50 transition-all duration-300 group"
              >
                <Icon className="text-accent text-3xl mb-4 group-hover:scale-110 transition-transform" />
                <h4 className="text-white font-bold mb-1">{method.label}</h4>
                <p className="text-accent text-sm font-semibold mb-3 break-all">{method.value}</p>
                <p className="text-white/50 text-xs">{method.description}</p>
              </motion.a>
            );
          })}
        </div>

        {/* Form and Info */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Contact Form */}
          <motion.div variants={itemVariants}>
            <h2 className="text-3xl font-bold text-white mb-8">Send me a message</h2>

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Name */}
              <div>
                <label className="block text-white font-semibold mb-2">Name</label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 rounded-lg bg-dark-tertiary border border-dark-tertiary focus:border-accent text-white placeholder-white/30 transition-all duration-300 focus:outline-none"
                  placeholder="Your name"
                />
              </div>

              {/* Email */}
              <div>
                <label className="block text-white font-semibold mb-2">Email</label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 rounded-lg bg-dark-tertiary border border-dark-tertiary focus:border-accent text-white placeholder-white/30 transition-all duration-300 focus:outline-none"
                  placeholder="your.email@example.com"
                />
              </div>

              {/* Subject */}
              <div>
                <label className="block text-white font-semibold mb-2">Subject</label>
                <input
                  type="text"
                  name="subject"
                  value={formData.subject}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 rounded-lg bg-dark-tertiary border border-dark-tertiary focus:border-accent text-white placeholder-white/30 transition-all duration-300 focus:outline-none"
                  placeholder="What's this about?"
                />
              </div>

              {/* Message */}
              <div>
                <label className="block text-white font-semibold mb-2">Message</label>
                <textarea
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  required
                  rows={5}
                  className="w-full px-4 py-3 rounded-lg bg-dark-tertiary border border-dark-tertiary focus:border-accent text-white placeholder-white/30 transition-all duration-300 focus:outline-none resize-none"
                  placeholder="Tell me more about your project..."
                />
              </div>

              {/* Submit Button */}
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="submit"
                className="w-full px-6 py-4 bg-accent text-dark font-bold rounded-lg hover:bg-accent-light transition-all duration-300"
              >
                {submitted ? '✓ Message Sent!' : 'Send Message'}
              </motion.button>
            </form>
          </motion.div>

          {/* Info */}
          <motion.div variants={itemVariants} className="flex flex-col justify-between">
            <div className="space-y-8">
              <div>
                <h3 className="text-2xl font-bold text-accent mb-4">Let's Collaborate</h3>
                <p className="text-white/70 text-lg leading-relaxed">
                  Whether you need a marketing strategist, content creator, or campaign manager,
                  I'm here to help bring your vision to life. I'm available for:
                </p>
              </div>

              <ul className="space-y-3">
                {[
                  'Full-time opportunities',
                  'Freelance projects',
                  'Contract roles',
                  'Consulting and collaborations',
                  'Speaking engagements',
                ].map((item, i) => (
                  <motion.li
                    key={i}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.1 }}
                    className="flex items-center gap-3 text-white/70"
                  >
                    <span className="w-2 h-2 rounded-full bg-accent" />
                    {item}
                  </motion.li>
                ))}
              </ul>

              {/* Location */}
              <motion.div
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                className="p-6 rounded-xl glass-effect border border-accent/30"
              >
                <div className="flex items-start gap-4">
                  <FiMapPin className="text-accent text-2xl flex-shrink-0 mt-1" />
                  <div>
                    <h4 className="text-white font-bold mb-1">Location</h4>
                    <p className="text-white/70">Delhi, India</p>
                    <p className="text-white/50 text-sm mt-2">Open to opportunities worldwide</p>
                  </div>
                </div>
              </motion.div>
            </div>

            {/* Response Time */}
            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              className="p-6 rounded-xl bg-gradient-accent"
            >
              <p className="text-white font-semibold mb-2">Response Time</p>
              <p className="text-white/80">I typically respond to inquiries within 24-48 hours</p>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </motion.section>
  );
}
