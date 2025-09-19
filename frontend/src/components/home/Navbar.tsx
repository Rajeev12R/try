import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

const Navbar: React.FC = () => {
  const [activeLink, setActiveLink] = useState('Home');
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const navLinks = ['Home', 'Ocean Data', 'Technology', 'Research'];

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      if (isMenuOpen) {
        setIsMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isMenuOpen]);

  return (
    <nav className="fixed top-2.5 left-1/2 flex w-full max-w-7xl -translate-x-1/2 items-center justify-between px-6 py-1.5 pr-4 md:top-4 z-50">
      {/* Logo */}
      <Link 
        className="hidden size-8 p-1 drop-shadow-xl delay-200 md:size-9 lg:block" 
        aria-label="Homepage" 
        to="/"
        onClick={(e) => {
          e.preventDefault();
          setActiveLink('Home');
        }}
      >
<svg className="size-[35px]" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 5350 5350">
  <path 
    className="fill-white dark:fill-black" 
    d="M2675 1345c-720 0-1305 585-1305 1305s585 1305 1305 1305 1305-585 1305-1305-585-1305-1305-1305zm-655 1730c-90 0-165-75-165-165s75-165 165-165 165 75 165 165-75 165-165 165zm655 0c-90 0-165-75-165-165s75-165 165-165 165 75 165 165-75 165-165 165zm655 0c-90 0-165-75-165-165s75-165 165-165 165 75 165 165-75 165-165 165z"
  />
</svg>
      </Link>

      {/* Navigation Links */}
      <div className="relative flex justify-center mx-auto">
        <ul className="relative flex min-h-14 items-center justify-center rounded-[22px] border border-black/10 bg-black/30 px-1 py-1 shadow-xl backdrop-blur-2xl dark:border-white/10 dark:bg-white/10">
          <div className="flex items-center" style={{ opacity: 1, filter: 'blur(0px)' }}>
            {navLinks.map((link) => (
              <li key={link} className="relative list-none">
                <Link
                  className={`block px-4 py-1.5 text-sm font-light transition ${
                    activeLink === link 
                      ? 'text-white' 
                      : 'text-white/70 hover:text-white dark:text-white/70'
                  }`}
                  to={`/${link.toLowerCase() === 'home' ? '' : link.toLowerCase()}`}
                  onClick={(e) => {
                    e.preventDefault();
                    setActiveLink(link);
                  }}
                >
                  {link}
                </Link>
                {activeLink === link && (
                  <span className="absolute inset-0 -z-10 w-full rounded-full bg-black/15 dark:bg-white/10">
                    <div className="bg-primary absolute -top-[9px] left-1/2 h-1 w-8 -translate-x-1/2 rounded-t-full">
                      <div className="bg-primary/20 absolute -top-2 -left-2 h-6 w-12 rounded-full blur-md"></div>
                      <div className="bg-primary/20 absolute -top-1 h-6 w-8 rounded-full blur-md"></div>
                      <div className="bg-primary/20 absolute top-0 left-2 h-4 w-4 rounded-full blur-sm"></div>
                    </div>
                  </span>
                )}
              </li>
            ))}
            
            {/* Book a Call Button */}
            <li className="ml-1 list-none">
              <button
                className="items-center justify-center gap-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive shadow-xs has-[>svg]:px-3 relative inline-block h-full cursor-pointer rounded-full bg-white/10 px-4 py-1.5 text-sm font-light whitespace-nowrap text-white transition-all duration-300 hover:bg-white/15 hover:text-white/90 dark:text-white/70"
                onClick={() => {
                  // Handle book a call action
                  console.log('Book a call clicked');
                }}
              >
                Get Access
                <div aria-hidden="true" className="absolute bottom-0 h-1/3 w-full -translate-x-4 rounded-full bg-white opacity-30 blur-sm"></div>
              </button>
            </li>
          </div>
        </ul>
      </div>

      {/* Mobile Menu Button */}
      <div className="hidden items-center gap-2 delay-200 lg:flex">
        <button
          className="inline-flex cursor-pointer items-center justify-center gap-2 whitespace-nowrap text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive hover:text-accent-foreground dark:hover:bg-accent/50 size-9 group rounded-2xl hover:bg-white/5"
          aria-expanded={isMenuOpen}
          aria-label="Toggle menu"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          <svg 
            className="pointer-events-none md:hidden size-6" 
            width="20" 
            height="20" 
            viewBox="0 0 24 24" 
            fill="none" 
            stroke="currentColor" 
            strokeWidth="2" 
            strokeLinecap="round" 
            strokeLinejoin="round"
          >
            <path 
              d="M4 12L20 12" 
              className={`origin-center -translate-y-[7px] transition-all duration-300 [transition-timing-function:cubic-bezier(.5,.85,.25,1.1)] ${
                isMenuOpen ? 'translate-x-0 translate-y-0 rotate-[315deg]' : ''
              }`}
            />
            <path 
              d="M4 12H20" 
              className={`origin-center transition-all duration-300 [transition-timing-function:cubic-bezier(.5,.85,.25,1.8)] ${
                isMenuOpen ? 'rotate-45' : ''
              }`}
            />
            <path 
              d="M4 12H20" 
              className={`origin-center translate-y-[7px] transition-all duration-300 [transition-timing-function:cubic-bezier(.5,.85,.25,1.1)] ${
                isMenuOpen ? 'translate-y-0 rotate-[135deg]' : ''
              }`}
            />
          </svg>
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            width="24" 
            height="24" 
            viewBox="0 0 24 24" 
            fill="none" 
            stroke="currentColor" 
            strokeWidth="2" 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            className="lucide lucide-command mx-0 hidden size-5 md:block" 
            aria-hidden="true"
          >
            <path d="M15 6v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3"></path>
          </svg>
        </button>
      </div>

      {/* Mobile Menu (when opened) */}
      {isMenuOpen && (
        <div className="absolute top-full right-0 mt-2 w-48 rounded-md bg-black/80 backdrop-blur-xl shadow-lg border border-white/10 py-2">
          {navLinks.map((link) => (
            <Link
              key={link}
              className={`block px-4 py-2 text-sm font-light transition ${
                activeLink === link 
                  ? 'text-white bg-white/10' 
                  : 'text-white/70 hover:text-white hover:bg-white/5'
              }`}
              to={`/${link.toLowerCase() === 'home' ? '' : link.toLowerCase()}`}
              onClick={(e) => {
                e.preventDefault();
                setActiveLink(link);
                setIsMenuOpen(false);
              }}
            >
              {link}
            </Link>
          ))}
          <div className="border-t border-white/10 my-2"></div>
          <button
            className="block w-full text-left px-4 py-2 text-sm font-light text-white/70 hover:text-white hover:bg-white/5"
            onClick={() => {
              // Handle book a call action
              console.log('Book a call clicked (mobile)');
              setIsMenuOpen(false);
            }}
          >
            Get Access
          </button>
        </div>
      )}
    </nav>
  );
};

export default Navbar;