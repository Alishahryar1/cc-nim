'use client';

import { motion } from 'framer-motion';

export default function CommunityContent() {
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

  const leadershipProgression = [
    {
      step: '1',
      title: 'Revived Inactive Chapter',
      description: 'Took ownership of a 2-year dormant community chapter and breathed new life into it.',
    },
    {
      step: '2',
      title: 'Built Team of 80+',
      description: 'Grew the community from zero active members to a thriving team of over 80 engaged students.',
    },
    {
      step: '3',
      title: 'Organized Events',
      description: 'Coordinated multiple events, workshops, and learning sessions for community growth.',
    },
    {
      step: '4',
      title: 'NGO Collaborations',
      description: 'Partnered with non-profits to create social impact and community service initiatives.',
    },
    {
      step: '5',
      title: 'Launched Podcasts',
      description: 'Created and launched community podcasts to amplify voices and share knowledge.',
    },
    {
      step: '6',
      title: 'Created Real Impact',
      description: 'Delivered measurable community impact through leadership, collaboration, and dedication.',
    },
  ];

  const initiatives = [
    {
      title: 'Community Engagement',
      metrics: ['80+ Members', '15+ Events', '1000+ Hours'],
      description: 'Built a thriving community through consistent engagement and meaningful initiatives.',
    },
    {
      title: 'Partnerships',
      metrics: ['5+ NGOs', '10+ Organizations', '20+ Projects'],
      description: 'Created strategic partnerships to amplify community impact and reach.',
    },
    {
      title: 'Learning Programs',
      metrics: ['12+ Workshops', '500+ Participants', '100% Positive Feedback'],
      description: 'Designed and delivered educational programs that empowered community members.',
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
        {/* Leadership Journey */}
        <motion.div variants={itemVariants} className="mb-20">
          <h2 className="text-4xl font-bold text-white text-center mb-16">
            Leadership <span className="gradient-text">in Action</span>
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {leadershipProgression.map((item, i) => (
              <motion.div
                key={i}
                variants={itemVariants}
                whileHover={{ y: -5 }}
                className="p-6 rounded-xl glass-effect border border-accent/30 hover:border-accent/60 transition-all duration-300"
              >
                <div className="flex items-start gap-4">
                  <motion.div
                    whileHover={{ scale: 1.2, rotate: 360 }}
                    transition={{ type: 'spring', stiffness: 200 }}
                    className="flex-shrink-0 w-10 h-10 rounded-full bg-accent flex items-center justify-center text-dark font-bold"
                  >
                    {item.step}
                  </motion.div>
                  <div>
                    <h4 className="text-white font-bold text-lg mb-2">{item.title}</h4>
                    <p className="text-white/60 text-sm leading-relaxed">{item.description}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Initiatives */}
        <motion.div variants={itemVariants} className="mb-20">
          <h3 className="text-3xl font-bold text-white text-center mb-12">
            Key <span className="gradient-text">Initiatives</span>
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {initiatives.map((initiative, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.2 }}
                className="p-8 rounded-2xl bg-gradient-accent"
              >
                <h4 className="text-2xl font-bold text-accent mb-6">{initiative.title}</h4>

                <div className="grid grid-cols-3 gap-4 mb-6">
                  {initiative.metrics.map((metric, j) => (
                    <div key={j} className="text-center">
                      <p className="text-white font-bold text-lg">{metric.split('+')[0]}+</p>
                      <p className="text-white/60 text-xs mt-1">{metric.split('+')[1]}</p>
                    </div>
                  ))}
                </div>

                <p className="text-white/80 text-sm leading-relaxed">{initiative.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Impact Section */}
        <motion.div
          variants={itemVariants}
          className="p-12 rounded-2xl glass-effect border border-accent/30 text-center"
        >
          <h3 className="text-3xl font-bold text-white mb-6">
            Community <span className="gradient-text">Leadership Philosophy</span>
          </h3>
          <p className="text-white/70 text-lg leading-relaxed max-w-3xl mx-auto">
            Leadership isn't about titles. It's about taking ownership, inspiring others, and creating
            meaningful impact. Every initiative I've led stems from a simple belief: when you invest in
            people and community, everyone wins. From reviving a dormant chapter to building a team of
            80+, my goal has always been to create spaces where people can learn, grow, and contribute.
          </p>
        </motion.div>
      </div>
    </motion.section>
  );
}
