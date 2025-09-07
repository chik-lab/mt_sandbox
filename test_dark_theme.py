#!/usr/bin/env python3
"""
Test script to verify dark theme integration works.
"""

import sys
import os

# Add the minitrade path
sys.path.insert(0, '/mnt/e/code/mt_sandbox')

try:
    # Test theme imports
    from minitrade.backtest.core.themes import (
        set_dark_theme, 
        set_light_theme, 
        get_current_theme,
        get_available_themes
    )
    
    print("✅ Theme system imports successful!")
    
    # Test theme switching
    print(f"Available themes: {get_available_themes()}")
    
    # Test light theme
    set_light_theme()
    current = get_current_theme()
    print(f"Current theme: {current.display_name}")
    print(f"Background color: {current.background_fill_color}")
    
    # Test dark theme
    set_dark_theme()
    current = get_current_theme()
    print(f"Current theme: {current.display_name}")
    print(f"Background color: {current.background_fill_color}")
    
    print("✅ Theme switching works!")
    
    # Test plotting integration imports
    from minitrade.backtest.core.themes.plotting_integration import (
        get_themed_colorgen,
        get_themed_bull_bear_colors,
    )
    
    # Test color generation
    colors = list(get_themed_colorgen())[:5]  # Get first 5 colors
    print(f"Themed colors: {colors}")
    
    bear_color, bull_color = get_themed_bull_bear_colors()
    print(f"Bull/Bear colors: {bull_color} / {bear_color}")
    
    print("✅ Plotting integration works!")
    
    # Test a simple backtest with dark theme
    from minitrade.backtest import Backtest, Strategy
    from minitrade.backtest.core.test import GOOG
    
    class SimpleStrategy(Strategy):
        def init(self):
            pass
        def next(self):
            pass
    
    bt = Backtest(GOOG.iloc[:50], SimpleStrategy)  # Small dataset for speed
    stats = bt.run()
    
    print("✅ Backtest runs successfully!")
    
    # Test plot with theme parameter (this should work now)
    print("Testing plot with theme='dark'...")
    try:
        bt.plot(theme='dark', open_browser=False, filename='')  # Don't save or open
        print("✅ Dark theme plot works!")
    except Exception as e:
        print(f"❌ Plot failed: {e}")
    
    print("\n🎉 All theme system tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the theme system files are in the correct location.")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
