/**
 * @file animations.js
 * @description Centralized Anime.js v4 animation suite for Second Brain.
 * Provides subtle, high-performance micro-interactions and transitions.
 */

import { animate, stagger } from 'animejs';

/**
 * Animate staggered entrance for cards in a grid
 */
export const animateCardsEntrance = (targets) => {
  if (!targets) return;
  const elements = typeof targets === 'string' 
    ? document.querySelectorAll(targets) 
    : targets;

  if (!elements || elements.length === 0) return;

  try {
    animate(elements, {
      opacity: [0, 1],
      translateY: [24, 0],
      scale: [0.96, 1],
      delay: stagger(45, { start: 30 }),
      duration: 480,
      ease: 'outExpo',
    });
  } catch (e) {
    console.warn('animateCardsEntrance error:', e);
  }
};

/**
 * Animate modal window entrance (spring scale + fade)
 */
export const animateModalEnter = (backdropEl, modalEl) => {
  try {
    if (backdropEl) {
      animate(backdropEl, {
        opacity: [0, 1],
        duration: 260,
        ease: 'outQuad',
      });
    }
    if (modalEl) {
      animate(modalEl, {
        opacity: [0, 1],
        translateY: [28, 0],
        scale: [0.94, 1],
        duration: 420,
        ease: 'outBack',
      });
    }
  } catch (e) {
    console.warn('animateModalEnter error:', e);
  }
};

/**
 * Smooth entrance for generated AI draft panel
 */
export const animateDraftReveal = (draftEl) => {
  if (!draftEl) return;
  try {
    animate(draftEl, {
      opacity: [0, 1],
      translateY: [18, 0],
      scale: [0.98, 1],
      duration: 420,
      ease: 'outCubic',
    });
  } catch (e) {
    console.warn('animateDraftReveal error:', e);
  }
};

/**
 * Number counter animation (e.g. score ring in Knowledge Analysis)
 */
export const animateNumber = (element, targetValue, duration = 1000) => {
  if (!element || typeof targetValue !== 'number') return;
  try {
    const counter = { val: 0 };
    animate(counter, {
      val: targetValue,
      duration,
      ease: 'outExpo',
      onUpdate: () => {
        if (element) {
          element.textContent = Math.round(counter.val);
        }
      },
    });
  } catch (e) {
    console.warn('animateNumber error:', e);
    if (element) element.textContent = targetValue;
  }
};

/**
 * Interactive button click pulse (subtle tactile feedback)
 */
export const animateButtonPress = (buttonEl) => {
  if (!buttonEl) return;
  try {
    animate(buttonEl, {
      scale: [0.93, 1],
      duration: 220,
      ease: 'outQuad',
    });
  } catch (e) {
    console.warn('animateButtonPress error:', e);
  }
};

/**
 * Chat message entrance animation
 */
export const animateChatMessageIn = (msgEl) => {
  if (!msgEl) return;
  try {
    animate(msgEl, {
      opacity: [0, 1],
      translateY: [14, 0],
      duration: 320,
      ease: 'outQuad',
    });
  } catch (e) {
    console.warn('animateChatMessageIn error:', e);
  }
};

/**
 * Seamless page tab transition (Notes <-> Chat)
 */
export const animatePageTransition = (pageEl) => {
  if (!pageEl) return;
  try {
    animate(pageEl, {
      opacity: [0.75, 1],
      translateY: [10, 0],
      duration: 280,
      ease: 'outQuad',
    });
  } catch (e) {
    console.warn('animatePageTransition error:', e);
  }
};

/**
 * Animate any list of elements with subtle stagger
 */
export const animateStaggeredList = (targets, staggerMs = 35) => {
  if (!targets) return;
  const elements = typeof targets === 'string'
    ? document.querySelectorAll(targets)
    : targets;
  if (!elements || elements.length === 0) return;

  try {
    animate(elements, {
      opacity: [0, 1],
      translateY: [12, 0],
      delay: stagger(staggerMs, { start: 20 }),
      duration: 360,
      ease: 'outQuad',
    });
  } catch (e) {
    console.warn('animateStaggeredList error:', e);
  }
};

/**
 * Subtle pulse animation for highlight/save feedback
 */
export const animatePulse = (targetEl) => {
  if (!targetEl) return;
  try {
    animate(targetEl, {
      scale: [1, 1.04, 1],
      duration: 320,
      ease: 'outQuad',
    });
  } catch (e) {
    console.warn('animatePulse error:', e);
  }
};

