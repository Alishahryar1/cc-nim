'use client';

import { motion } from 'framer-motion';

export default function AboutContent() {
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

  const philosophies = [
    {
      title: 'Story-Driven',
      description: 'Every brand has a story. My role is to find it, craft it, and amplify it in ways that resonate.',
    },
    {
      title: 'Strategy First',
      description: 'Creativity without strategy is noise. I combine data-driven insights with creative thinking.',
    },
    {
      title: 'Execution Focused',
      description: 'Ideas are just the beginning. I take ownership from concept to campaign completion.',
    },
    {
      title: 'People Centered',
      description: 'Marketing is about connection. Understanding people is the foundation of everything I do.',
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
        {/* Main Philosophy */}
        <motion.div variants={itemVariants} className="max-w-3xl mx-auto text-center mb-20">
          <p className="text-2xl text-white/80 leading-relaxed">
            I believe great marketing begins with a great story. My work brings together
            <span className="gradient-text font-bold"> strategy, content, creators, and campaigns</span>
            to help brands create experiences people remember.
          </p>
        </motion.div>

        {/* Philosophy Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-20">
          {philosophies.map((philosophy, i) => (
            <motion.div
              key={i}
              variants={itemVariants}
              className="p-8 rounded-xl glass-effect hover:border-accent/50 transition-all duration-300"
            >
              <h3 className="text-2xl font-bold text-accent mb-4">{philosophy.title}</h3>
              <p className="text-white/70 leading-relaxed">{philosophy.description}</p>
            </motion.div>
          ))}
        </div>

        {/* Approach */}
        <motion.div
          variants={itemVariants}
          className="max-w-3xl mx-auto bg-gradient-accent rounded-2xl p-12 text-center"
        >
          <h3 className="text-3xl font-bold text-white mb-4">My Approach</h3>
          <p className="text-white/80 leading-relaxed mb-6">
            I don't just create content or run campaigns. I craft narratives that educate, engage,
            and convert. Behind every script, email, or ad copy is a process of research, 
            experimentation, and refinement.
          </p>
          <p className="text-white/70 text-sm">
            The result? Marketing that feels effortless because it's been carefully thought through.
          </p>
        </motion.div>
      </div>
    </motion.section>
  );
}
