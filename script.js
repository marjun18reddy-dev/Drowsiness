/* ==========================================================================
   DROWSINESS DETECTION CNN — PRESENTATION LOGIC
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const slides = Array.from(document.querySelectorAll('.slide'));
  const totalSlides = slides.length;
  const progressBar = document.getElementById('progressBar');
  const currentSlideNum = document.getElementById('currentSlideNum');
  const totalSlideNum = document.getElementById('totalSlideNum');
  const btnPrev = document.getElementById('btnPrev');
  const btnNext = document.getElementById('btnNext');
  const btnMenu = document.getElementById('btnMenu');
  const btnCloseMenu = document.getElementById('btnCloseMenu');
  const menuDrawer = document.getElementById('menuDrawer');
  const menuList = document.getElementById('menuList');
  const btnFullscreen = document.getElementById('btnFullscreen');

  let currentIndex = 0;

  // Initialize total slide display
  totalSlideNum.textContent = totalSlides < 10 ? `0${totalSlides}` : totalSlides;

  // Build outline menu drawer items
  slides.forEach((slide, idx) => {
    const title = slide.getAttribute('data-title') || `Slide ${idx + 1}`;
    const numStr = (idx + 1) < 10 ? `0${idx + 1}` : `${idx + 1}`;
    
    const menuItem = document.createElement('div');
    menuItem.className = `menu-item ${idx === 0 ? 'active' : ''}`;
    menuItem.innerHTML = `<span class="menu-num">${numStr}</span> <span>${title}</span>`;
    menuItem.addEventListener('click', () => {
      goToSlide(idx);
      closeMenu();
    });
    menuList.appendChild(menuItem);
  });

  // Slide navigation function
  function goToSlide(index) {
    if (index < 0) index = 0;
    if (index >= totalSlides) index = totalSlides - 1;

    slides[currentIndex].classList.remove('active');
    currentIndex = index;
    slides[currentIndex].classList.add('active');

    // Update Progress Bar & Counter
    const progressPct = ((currentIndex + 1) / totalSlides) * 100;
    progressBar.style.width = `${progressPct}%`;
    
    const currentNumStr = (currentIndex + 1) < 10 ? `0${currentIndex + 1}` : `${currentIndex + 1}`;
    currentSlideNum.textContent = currentNumStr;

    // Update Drawer Active Item
    const menuItems = menuList.querySelectorAll('.menu-item');
    menuItems.forEach((item, idx) => {
      if (idx === currentIndex) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });

    // Scroll slide content to top
    const activeContent = slides[currentIndex].querySelector('.slide-content');
    if (activeContent) {
      activeContent.scrollTop = 0;
    }
  }

  // Event Listeners for Nav Buttons
  btnPrev.addEventListener('click', () => goToSlide(currentIndex - 1));
  btnNext.addEventListener('click', () => goToSlide(currentIndex + 1));

  // Menu Drawer handlers
  function openMenu() {
    menuDrawer.classList.add('open');
  }

  function closeMenu() {
    menuDrawer.classList.remove('open');
  }

  btnMenu.addEventListener('click', () => {
    if (menuDrawer.classList.contains('open')) {
      closeMenu();
    } else {
      openMenu();
    }
  });

  btnCloseMenu.addEventListener('click', closeMenu);

  // Keyboard navigation
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
      case ' ':
      case 'PageDown':
        e.preventDefault();
        goToSlide(currentIndex + 1);
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
      case 'PageUp':
        e.preventDefault();
        goToSlide(currentIndex - 1);
        break;
      case 'Home':
        e.preventDefault();
        goToSlide(0);
        break;
      case 'End':
        e.preventDefault();
        goToSlide(totalSlides - 1);
        break;
      case 'f':
      case 'F':
        e.preventDefault();
        toggleFullscreen();
        break;
      case 'Escape':
        closeMenu();
        break;
    }
  });

  // Touch Swipe Support
  let touchStartX = 0;
  let touchEndX = 0;

  document.addEventListener('touchstart', (e) => {
    touchStartX = e.changedTouches[0].screenX;
  }, { passive: true });

  document.addEventListener('touchend', (e) => {
    touchEndX = e.changedTouches[0].screenX;
    handleSwipe();
  }, { passive: true });

  function handleSwipe() {
    const swipeThreshold = 50;
    if (touchEndX < touchStartX - swipeThreshold) {
      goToSlide(currentIndex + 1); // Swipe left -> Next
    } else if (touchEndX > touchStartX + swipeThreshold) {
      goToSlide(currentIndex - 1); // Swipe right -> Prev
    }
  }

  // Fullscreen API
  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => {
        console.warn(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  }

  btnFullscreen.addEventListener('click', toggleFullscreen);

  // Initialize Slide 0
  goToSlide(0);

  // ==========================================================================
  // CHART.JS INITIALIZATION (TRAINING & VALIDATION METRICS)
  // ==========================================================================
  const epochs = ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5', 'Epoch 6', 'Epoch 7', 'Epoch 8', 'Epoch 9', 'Epoch 10', 'Epoch 11', 'Epoch 12', 'Epoch 13'];
  const trainLosses = [0.5677, 0.5224, 0.4958, 0.4722, 0.4560, 0.4564, 0.4519, 0.4491, 0.4333, 0.4275, 0.4253, 0.4186, 0.4161];
  const valLosses   = [0.5624, 1.1027, 0.4961, 0.4643, 0.4420, 1.9640, 0.4759, 0.5676, 0.4277, 0.4382, 0.4393, 0.4769, 0.4342];

  const trainAccs = [72.09, 75.97, 76.17, 76.75, 77.27, 76.19, 76.85, 76.17, 77.43, 77.96, 77.62, 78.60, 78.16];
  const valAccs   = [70.76, 60.89, 76.17, 74.27, 76.24, 50.15, 71.20, 71.05, 75.44, 76.17, 75.29, 74.63, 76.83];

  // Chart configuration defaults
  if (typeof Chart !== 'undefined') {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = 'Inter, sans-serif';

    // 1. Loss Chart
    const ctxLoss = document.getElementById('lossChart').getContext('2d');
    new Chart(ctxLoss, {
      type: 'line',
      data: {
        labels: epochs,
        datasets: [
          {
            label: 'Training Loss',
            data: trainLosses,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2.5,
            pointRadius: 4,
            pointBackgroundColor: '#3b82f6',
            tension: 0.3,
            fill: true
          },
          {
            label: 'Validation Loss',
            data: valLosses,
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.1)',
            borderWidth: 2.5,
            pointRadius: (ctx) => (ctx.dataIndex === 8 ? 8 : 4), // Highlight Epoch 9 best loss
            pointBackgroundColor: (ctx) => (ctx.dataIndex === 8 ? '#10b981' : '#06b6d4'),
            tension: 0.3,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { boxWidth: 12, font: { weight: '600' } }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              footer: (items) => {
                const idx = items[0].dataIndex;
                if (idx === 8) return '★ Best Val Loss: 0.4277';
                if (idx === 12) return '⚡ Early Stopping Triggered';
                return '';
              }
            }
          }
        },
        scales: {
          x: { grid: { color: 'rgba(255, 255, 255, 0.05)' } },
          y: { 
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            title: { display: true, text: 'Loss', color: '#64748b' }
          }
        }
      }
    });

    // 2. Accuracy Chart
    const ctxAcc = document.getElementById('accuracyChart').getContext('2d');
    new Chart(ctxAcc, {
      type: 'line',
      data: {
        labels: epochs,
        datasets: [
          {
            label: 'Training Accuracy (%)',
            data: trainAccs,
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            borderWidth: 2.5,
            pointRadius: 4,
            pointBackgroundColor: '#8b5cf6',
            tension: 0.3,
            fill: true
          },
          {
            label: 'Validation Accuracy (%)',
            data: valAccs,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 2.5,
            pointRadius: (ctx) => (ctx.dataIndex === 12 ? 8 : 4), // Highlight Epoch 13 peak acc
            pointBackgroundColor: (ctx) => (ctx.dataIndex === 12 ? '#f59e0b' : '#10b981'),
            tension: 0.3,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { boxWidth: 12, font: { weight: '600' } }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              footer: (items) => {
                const idx = items[0].dataIndex;
                if (idx === 12) return '★ Peak Val Acc: 76.83%';
                return '';
              }
            }
          }
        },
        scales: {
          x: { grid: { color: 'rgba(255, 255, 255, 0.05)' } },
          y: { 
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            title: { display: true, text: 'Accuracy (%)', color: '#64748b' },
            min: 45,
            max: 85
          }
        }
      }
    });
  }
});
