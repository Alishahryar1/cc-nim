'use client';

import { motion } from 'framer-motion';

export default function ExperienceContent() {
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
    hidden: { opacity: 0, x: -30 },
    visible: {
      opacity: 1,
      x: 0,
      transition: { duration: 0.8 },
    },
  };

  const experiences = [
    {
      period: '2024-Present',
      company: 'Arviend Sud',
      title: 'Content Strategist & Scriptwriter',
      description:
        'Collaborating with a personal brand followed by millions. Transforming ideas into content that educates, engages, and converts across all platforms.',
      achievements: ['60+ Scripts Created', '20+ Email Campaigns', '10+ WhatsApp Campaigns', '150K+ Reach'],
      skills: ['Scriptwriting', 'Content Strategy', 'Copy Writing', 'Campaign Planning', 'Analytics'],
    },
    {
      period: '2023-2024',
      company: 'Various Brands',
      title: 'Marketing & Advertising Specialist',
      description:
        'Managed marketing campaigns, created advertising strategies, and coordinated with creators and influencers for brand partnerships.',
      achievements: ['5+ Brand Campaigns', 'Creator Partnerships', 'Viral Content', 'High Engagement Rates'],
      skills: ['Brand Marketing', 'Influencer Marketing', 'Campaign Management', 'Creative Direction'],
    },
    {
      period: '2022-2023',
      company: 'Community Leadership',
      title: 'Secretary & Organization Lead',
      description:
        'Revived an inactive 2-year chapter by building a team of 80+ students, organizing events, managing NGO collaborations, and launching community initiatives.',
      achievements: ['Built Team of 80+', 'Multiple Events Organized', 'NGO Collaborations', 'Community Impact'],
      skills: ['Leadership', 'Team Building', 'Event Management', 'Community Engagement', 'Project Coordination'],
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
        {/* Experience Timeline */}
        <div className="max-w-4xl mx-auto">
          {experiences.map((exp, i) => (
            <motion.div key={i} variants={itemVariants} className="mb-12 relative">
              {/* Timeline connector */}
              {i !== experiences.length - 1 && (
                <div className="absolute left-8 top-24 bottom-0 w-1 bg-gradient-to-b from-accent to-accent/30" />
              )}

              {/* Timeline dot */}
              <div className="absolute left-0 top-0 w-16 h-16 flex items-center justify-center">
                <motion.div
                  whileHover={{ scale: 1.3 }}
                  className="w-12 h-12 rounded-full bg-accent flex items-center justify-center"
                >
                  <span className="text-dark font-bold text-lg">✓</span>
                </motion.div>
              </div>

              {/* Content */}
              <div className="pl-32 pb-8">
                <p className="text-accent text-sm font-bold uppercase tracking-wider mb-2">
                  {exp.period}
                </p>
                <h3 className="text-3xl font-bold text-white mb-1">{exp.title}</h3>
                <p className="text-accent-light text-lg font-semibold mb-4">{exp.company}</p>
                <p className="text-white/70 text-lg leading-relaxed mb-6">{exp.description}</p>

                {/* Achievements */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  {exp.achievements.map((achievement, j) => (
                    <motion.div
                      key={j}
                      initial={{ opacity: 0, y: 10 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: j * 0.1 }}
                      className="p-3 rounded-lg bg-dark-tertiary"
                    >
                      <p className="text-accent text-xs font-bold">{achievement}</p>
                    </motion.div>
                  ))}
                </div>

                {/* Skills */}
                <div className="flex flex-wrap gap-2">
                  {exp.skills.map((skill, j) => (
                    <motion.span
                      key={j}
                      initial={{ opacity: 0, scale: 0.8 }}
                      whileInView={{ opacity: 1, scale: 1 }}
                      viewport={{ once: true }}
                      transition={{ delay: j * 0.05 }}
                      className="px-3 py-1 rounded-full bg-accent/10 text-accent text-xs font-semibold"
                    >
                      {skill}
                    </motion.span>
                  ))}
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Skills Summary */}
        <motion.div
          variants={itemVariants}
          className="max-w-4xl mx-auto mt-20 p-12 rounded-2xl glass-effect border border-accent/30"
        >
          <h3 className="text-3xl font-bold text-white mb-6">Core Competencies</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[
              { title: 'Marketing', items: ['Strategy', 'Brand Building', 'Campaigns', 'Content Planning'] },
              { title: 'Storytelling', items: ['Copywriting', 'Scriptwriting', 'Narrative Design', 'Messaging'] },
              { title: 'Execution', items: ['Project Management', 'Team Coordination', 'Event Planning', 'Launch Management'] },
              { title: 'Leadership', items: ['Team Building', 'Community Management', 'Mentoring', 'Decision Making'] },
            ].map((competency, i) => (
              <motion.div key={i} initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
                <h4 className="text-accent font-bold mb-3">{competency.title}</h4>
                <ul className="space-y-2">
                  {competency.items.map((item, j) => (
                    <li key={j} className="text-white/70 text-sm flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-accent" />
                      {item}
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </motion.section>
  );
}
