#!/usr/bin/env python3
"""
MACE Banner and Credits Display
"""

def get_mace_banner():
    """Return MACE ASCII art banner"""
    banner = r"""
      ███╗   ███╗ █████╗  ██████╗███████╗
      ████╗ ████║██╔══██╗██╔════╝██╔════╝
      ██╔████╔██║███████║██║     █████╗  
      ██║╚██╔╝██║██╔══██║██║     ██╔══╝  
      ██║ ╚═╝ ██║██║  ██║╚██████╗███████╗
      ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝
    """
    return banner

def get_mace_banner_alt():
    """Alternative banner with subtitle"""
    banner = r"""
     __  __    _    ____ _____ 
    |  \/  |  / \  / ___| ____|
    | |\/| | / _ \| |   |  _|  
    | |  | |/ ___ \ |___| |___ 
    |_|  |_/_/   \_\____|_____|
    
    Mendoza Automated CRYSTAL Engine
    """
    return banner

def get_mace_banner_compact():
    """Compact banner for frequent use"""
    banner = r"""
    ╔═╗╔═╗╔═╗╔═╗
    ║║║╠═╣║  ╠╣ 
    ╩ ╩╩ ╩╚═╝╚═╝ v1.0
    """
    return banner

def get_credits():
    """Return formatted credits"""
    credits = """
============================================================
Developed at Michigan State University
Mendoza Group - Materials Science & Engineering

Primary Developer:  Marcus Djokic (PhD Student)
Contributors:       Daniel Maldonado Lopez (PhD Student)
                    Brandon Lewis (Undergraduate)
                    Dr. William Comaskey (Postdoc)
PI:                 Prof. Jose Luis Mendoza-Cortes
============================================================
    """
    return credits

def get_full_banner():
    """Return full banner with credits"""
    return get_mace_banner() + "\n" + get_credits()

def print_banner(style='full'):
    """Print MACE banner
    
    Args:
        style: 'full', 'banner', 'alt', 'compact', or 'credits'
    """
    if style == 'full':
        print(get_full_banner())
    elif style == 'banner':
        print(get_mace_banner())
    elif style == 'alt':
        print(get_mace_banner_alt())
    elif style == 'compact':
        print(get_mace_banner_compact())
    elif style == 'credits':
        print(get_credits())

if __name__ == '__main__':
    import sys
    style = sys.argv[1] if len(sys.argv) > 1 else 'full'
    print_banner(style)