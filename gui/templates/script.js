(function() {
    // 缓存常用DOM元素
    const elements = {
        container: document.querySelector('.scroll-container'), // 主容器
        inputSection: document.querySelector('.input-section'), // 输入部分
        capabilitiesSection: document.querySelector('.capabilities-section'), // 功能部分
        body: document.body,
        html: document.documentElement,
        form: document.getElementById('textAnalysisForm')
    };

    // 状态控制
    let isAnimating = false;
    let currentSection = 'input';
    let startY = 0; // 用于触摸事件
    let isTouchDevice = 'ontouchstart' in window; // 检测触摸设备

    function init() {
        elements.body.style.overflow = 'hidden';

        // 统一使用transform实现动画
        elements.capabilitiesSection.style.transform = 'translateY(100%)';

        // 桌面端使用滚轮事件
        window.addEventListener('wheel', throttle(handleScroll, 500), { passive: false });

        // 移动端使用触摸事件
        if (isTouchDevice) {
            setupTouchEvents();
        }

        // 防止表单区域滚动触发页面切换
        if (elements.form) {
            elements.form.addEventListener('wheel', (e) => e.stopPropagation(), { passive: false });
        }
    }

    function setupTouchEvents() {
        // 触摸开始 - 记录起始位置
        elements.container.addEventListener('touchstart', (e) => {
            startY = e.touches[0].clientY;
        }, { passive: true });

        // 触摸移动 - 处理滑动
        elements.container.addEventListener('touchmove', (e) => {
            if (isAnimating) {
                e.preventDefault();
                return;
            }

            const y = e.touches[0].clientY;
            const deltaY = y - startY;

            // 只有垂直滑动超过阈值才处理
            if (Math.abs(deltaY) > 10) {
                e.preventDefault();

                // 实时跟随手指移动效果
                if (currentSection === 'input' && deltaY < 0) {
                    // 向下滑动 - 切换到功能页
                    const progress = Math.min(1, Math.abs(deltaY) / 100);
                    applyTransform(progress);
                } else if (currentSection === 'capabilities' && deltaY > 0) {
                    // 向上滑动 - 返回输入页
                    const progress = Math.min(1, deltaY / 100);
                    applyTransform(1 - progress);
                }
            }
        }, { passive: false });

        // 触摸结束 - 完成切换或回弹
        elements.container.addEventListener('touchend', (e) => {
            const y = e.changedTouches[0].clientY;
            const deltaY = y - startY;

            // 滑动距离超过阈值才切换，否则回弹
            if (Math.abs(deltaY) > 50) {
                if (deltaY < 0 && currentSection === 'input') {
                    switchToCapabilities();
                } else if (deltaY > 0 && currentSection === 'capabilities') {
                    switchToInput();
                } else {
                    resetTransform();
                }
            } else {
                resetTransform();
            }
        }, { passive: true });
    }

    // 应用实时变换效果
    function applyTransform(progress) {
        elements.inputSection.style.transition = 'none';
        elements.capabilitiesSection.style.transition = 'none';

        elements.inputSection.style.transform = `translateY(${-100 * progress}%)`;
        elements.capabilitiesSection.style.transform = `translateY(${100 - 100 * progress}%)`;
    }

    // 重置变换
    function resetTransform() {
        isAnimating = true;

        elements.inputSection.style.transition = 'transform 0.3s ease';
        elements.capabilitiesSection.style.transition = 'transform 0.3s ease';

        if (currentSection === 'input') {
            elements.inputSection.style.transform = 'translateY(0)';
            elements.capabilitiesSection.style.transform = 'translateY(100%)';
        } else {
            elements.inputSection.style.transform = 'translateY(-100%)';
            elements.capabilitiesSection.style.transform = 'translateY(0)';
        }

        setTimeout(() => {
            isAnimating = false;
            elements.inputSection.style.transition = '';
            elements.capabilitiesSection.style.transition = '';
        }, 300);
    }

    function handleScroll(e) {
        if (isAnimating) return;
        e.preventDefault();

        const isScrollingDown = e.deltaY > 0;

        if (isScrollingDown && currentSection === 'input') {
            switchToCapabilities();
        } else if (!isScrollingDown && currentSection === 'capabilities') {
            switchToInput();
        }
    }

    function switchToCapabilities() {
        isAnimating = true;
        currentSection = 'capabilities';

        elements.inputSection.style.transition = 'transform 0.5s cubic-bezier(0.25, 0.1, 0.25, 1)';
        elements.capabilitiesSection.style.transition = 'transform 0.5s cubic-bezier(0.25, 0.1, 0.25, 1)';

        elements.inputSection.style.transform = 'translateY(-100%)';
        elements.capabilitiesSection.style.transform = 'translateY(0)';

        setTimeout(() => {
            isAnimating = false;
        }, 500);
    }

    function switchToInput() {
        isAnimating = true;
        currentSection = 'input';

        elements.inputSection.style.transition = 'transform 0.5s cubic-bezier(0.25, 0.1, 0.25, 1)';
        elements.capabilitiesSection.style.transition = 'transform 0.5s cubic-bezier(0.25, 0.1, 0.25, 1)';

        elements.inputSection.style.transform = 'translateY(0)';
        elements.capabilitiesSection.style.transform = 'translateY(100%)';

        setTimeout(() => {
            isAnimating = false;
        }, 500);
    }

    // 节流函数
    function throttle(fn, wait) {
        let lastTime = 0;
        return function(...args) {
            const now = Date.now();
            if (now - lastTime >= wait) {
                fn.apply(this, args);
                lastTime = now;
            }
        };
    }

    document.addEventListener('DOMContentLoaded', init);
})();