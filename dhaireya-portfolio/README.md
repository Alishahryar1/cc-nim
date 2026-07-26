# Dhaireya Jagya - Marketing Portfolio

A modern, animated portfolio website showcasing marketing expertise, creative work, and community leadership.

## 🚀 Features

- **Modern Design**: Clean, editorial-style design with smooth animations
- **Multi-Page Structure**: Home, About, Work, Experience, Community, and Contact pages
- **Framer Motion Animations**: Sophisticated animations and transitions throughout
- **Interactive Elements**: 
  - Logo carousel with auto-rotation
  - Work showcase with detailed view switching
  - Timeline-based experience display
  - Responsive contact form
- **Mobile Responsive**: Fully optimized for all device sizes
- **Dark Theme**: Premium dark theme with accent colors
- **Performance Optimized**: Built with Next.js for optimal performance

## 📁 Project Structure

```
dhaireya-portfolio/
├── app/
│   ├── layout.tsx           # Root layout
│   ├── page.tsx             # Home page
│   ├── globals.css          # Global styles
│   ├── about/page.tsx       # About page
│   ├── work/page.tsx        # Work showcase page
│   ├── experience/page.tsx  # Experience page
│   ├── community/page.tsx   # Community page
│   └── contact/page.tsx     # Contact page
├── components/
│   ├── Navigation.tsx       # Navigation bar
│   ├── Footer.tsx           # Footer
│   ├── PageHeader.tsx       # Page header component
│   ├── HeroSection.tsx      # Hero section
│   ├── LogoShowcase.tsx     # Logo carousel
│   ├── AboutPreview.tsx     # About preview
│   ├── WorkShowcase.tsx     # Work showcase
│   ├── ExperiencePreview.tsx # Experience preview
│   ├── CTASection.tsx       # Call-to-action
│   └── sections/            # Page content sections
│       ├── AboutContent.tsx
│       ├── WorkContent.tsx
│       ├── ExperienceContent.tsx
│       ├── CommunityContent.tsx
│       └── ContactContent.tsx
├── package.json             # Dependencies
├── tsconfig.json            # TypeScript config
├── tailwind.config.js       # Tailwind CSS config
├── postcss.config.js        # PostCSS config
└── next.config.js           # Next.js config
```

## 🛠️ Tech Stack

- **Framework**: Next.js 14
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Icons**: React Icons
- **Font**: Poppins, Playfair Display, Inter

## 📦 Installation

1. Navigate to the project directory:
```bash
cd dhaireya-portfolio
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## 🎨 Customization

### Colors
Update the theme colors in `tailwind.config.js`:
```js
colors: {
  accent: '#ff6b35',
  'accent-light': '#ff8c5a',
  dark: '#0f0f0f',
  // ...
}
```

### Content
Update portfolio content in:
- `components/HeroSection.tsx` - Hero text and CTAs
- `components/sections/` - Individual page content
- `components/LogoShowcase.tsx` - Brand logos/collaborations
- `components/sections/WorkContent.tsx` - Work samples and links

### Fonts
Fonts are imported from Google Fonts in `app/globals.css`:
```css
@import url('https://fonts.googleapis.com/css2?family=Poppins:...');
```

## 🚀 Deployment

### Vercel (Recommended)
1. Push your code to GitHub
2. Connect your repo to Vercel
3. Deploy with one click

### Build for Production
```bash
npm run build
npm start
```

## 📝 Key Features Explained

### 1. **Hero Section**
- Eye-catching headline with gradient text
- Animated background elements
- Primary and secondary CTA buttons
- Scroll indicator

### 2. **Logo Showcase**
- Auto-rotating carousel (3-second interval)
- Interactive indicators
- Click to switch logos
- Grid display below carousel

### 3. **Work Showcase**
- Featured work display with detailed view
- Interactive selection between work items
- Content types, reach, and direct links
- Smooth transitions between items

### 4. **Experience Timeline**
- Visual timeline with animated dots
- Experience cards with achievements
- Skills display
- Competencies summary section

### 5. **Contact Form**
- Multiple contact methods
- Interactive form with validation
- Success feedback message
- Direct contact links (email, phone, socials)

## 🎯 Pages Overview

- **Home**: Hero, logos, about preview, work showcase, experience preview, CTA
- **About**: Philosophy, approach, core beliefs
- **Work**: Detailed work showcase with filterable items
- **Experience**: Full timeline, achievements, competencies
- **Community**: Leadership progression, initiatives, impact
- **Contact**: Contact form, multiple contact methods, response time info

## 📱 Responsive Design

The portfolio is fully responsive with breakpoints:
- Mobile: < 768px
- Tablet: 768px - 1024px
- Desktop: > 1024px

## ⚡ Performance

- Image optimization
- Code splitting
- Lazy loading
- CSS optimization
- Font optimization

## 🔗 Important Links

- **Email**: jagya18dhaireya18@gmail.com
- **Phone**: +91 8448922579
- **LinkedIn**: https://www.linkedin.com/in/dhaireya-jagya-298498272
- **Instagram**: https://www.instagram.com/arviendsud

## 📄 License

This portfolio is custom-built for Dhaireya Jagya. All content and design are proprietary.

## 🤝 Support

For any questions or modifications, please reach out via the contact form or direct email.

---

Built with ❤️ using Next.js, React, and Tailwind CSS
