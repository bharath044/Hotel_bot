# hotel_bot_final_no_gsheet_with_tables.py
"""
Friends Catering - Final Bot (NO GOOGLE SHEET)
- Banner + Specials + Categories
- Quantity buttons 1️⃣..5️⃣
- Continue Order / Go to Payment
- Search 🔎
- Ask TABLE (1..10) after name
- Ask phone before UPI payment
- UPI QR → Screenshot → OCR → Groq/local verify
"""

import os
import re
import json
import logging
import datetime
from typing import Optional, List, Tuple, Dict
from pprint import pformat

import requests
import difflib

from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import pytesseract
import cv2
import numpy as np

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# ---------------- CONFIG ----------------
TOKEN = "8583512591:AAE4OthAdzV3t2LFSjsrRWJBGYo21BEsRc8"  # set your bot token

BANNER_IMAGE_PATH = "ss.jpeg"   # optional banner
QR_IMAGE_PATH = "image.png"    # your UPI QR image path

YOUR_UPI_ID = "bharathkaleeswaran004@okicici"

GROQ_API_KEY = "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"
GROQ_MODEL = "gemma-2-7b-instruct"

ADMIN_CHAT_ID = None
ITEMS_PER_PAGE = 6
AMOUNT_TOLERANCE = 2.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("friends_catering")

# ---------------- TESSERACT PATH (Windows common) ----------------
_windows_tess_paths = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]
for p in _windows_tess_paths:
    if os.path.exists(p):
        pytesseract.pytesseract.tesseract_cmd = p
        logger.info("Using tesseract at %s", p)
        break

# ---------------- MENU & CATEGORIES ----------------
MENU: Dict[str, int] = {
    # Biriyani
    "Hyderabadi Biriyani": 120,  "Veg Biriyani": 80,
    "Biriyani Rice": 60, "Gee Rice": 60, "Jeera Rice": 60,
    "White Rice Meals": 60, "Boil Rice Meals": 60,
    # Fried Rice
    "Veg Fried Rice": 70,
    "Gobi Fried Rice": 70, "Paneer Fried Rice": 80, "Mushroom Fried Rice": 80,
    # Noodles & Gobi
    "Veg Noodles": 60, "Paneer Noodles": 80,
    "Gobi Noodles": 70, "Gobi Manchurian": 60, "Gobi Chilly": 60, 
   
    # Veg North Indian
    "Paneer Butter Masala": 80, "Sayi Paneer": 80, "Kema Paneer": 80, "Paneer Burji": 80,
    "Palak Paneer": 80, "Mutter Paneer": 80, "Paneer Chilly": 80, "Paneer Manchurian": 80,
    "Dal Fry Tadka": 80, "Dal Fry": 80, "Green Peas Masala": 80, "Channa Masala": 80,
    "Veg Masala": 80, "Rajma Masala": 80, "Baddee Masala": 80, "Bagda Masala": 80,
    "Capsicum Masala": 80, "Egg Masala": 80, "Mushroom Fry": 80, "Mix Sabji": 80,
    "Aloo Gobi Sabji": 80,
    # Specials
    "Pappam": 60, "Puttu Kadalacurry": 70, "Paniyaram": 70, "Panipuri": 40,
    "Ghee Roast": 120, "Idiyappam": 70, "Pav Bhaji": 60
}

CATEGORIES = {
    "Biriyani": ["Hyderabadi Biriyani",  "Veg Biriyani",
                 "Biriyani Rice", "Gee Rice", "Jeera Rice", "White Rice Meals", "Boil Rice Meals"],
    "Fried Rice": [ "Veg Fried Rice", "Gobi Fried Rice", "Paneer Fried Rice", "Mushroom Fried Rice"],
    "Noodles & Gobi": [ "Veg Noodles", "Paneer Noodles", "Gobi Noodles", "Gobi Manchurian", "Gobi Chilly" ],
    
    "Veg North Indian": ["Paneer Butter Masala", "Sayi Paneer", "Kema Paneer", "Paneer Burji", "Palak Paneer",
                         "Mutter Paneer", "Paneer Chilly", "Paneer Manchurian", "Dal Fry Tadka", "Dal Fry",
                         "Green Peas Masala", "Channa Masala", "Veg Masala", "Rajma Masala", "Baddee Masala",
                         "Bagda Masala", "Capsicum Masala", "Egg Masala", "Mushroom Fry", "Mix Sabji", "Aloo Gobi Sabji"],
    "Specials": ["Aappam", "Puttu Kadalacurry", "Paniyaram", "Panipuri", "Ghee Roast", "Idiyappam", "Pav Bhaji"]
}

# ---------------- OCR helpers ----------------
def ocr_pass_gray_resize(path: str, scale: float = 1.0, contrast: float = 1.6) -> str:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return ""
    if scale != 1.0:
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    gray = ImageOps.grayscale(img)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    if contrast != 1.0:
        gray = ImageEnhance.Contrast(gray).enhance(contrast)
    try:
        return pytesseract.image_to_string(gray)
    except Exception:
        return ""

def ocr_pass_adaptive_thresh(path: str, scale: float = 1.0) -> str:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    h, w = img.shape[:2]
    if scale != 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 25, 10)
    tmp = "/tmp/_ocr_adaptive.png"
    cv2.imwrite(tmp, th)
    try:
        return pytesseract.image_to_string(Image.open(tmp))
    except Exception:
        return ""

def ocr_multi_pass(path: str) -> Tuple[str, List[Tuple[str,int]]]:
    attempts = []
    attempts.append(ocr_pass_gray_resize(path, scale=1.0, contrast=1.6))
    attempts.append(ocr_pass_gray_resize(path, scale=1.5, contrast=1.8))
    attempts.append(ocr_pass_gray_resize(path, scale=2.0, contrast=2.0))
    attempts.append(ocr_pass_adaptive_thresh(path, scale=1.2))
    attempts.append(ocr_pass_adaptive_thresh(path, scale=1.5))
    scored = []
    for t in attempts:
        if not t:
            scored.append((t, 0))
            continue
        score = len(re.findall(r"\d", t))
        score += len(re.findall(r"₹|Rs|INR|paid|Paid|successful|Paid to|Paid successfully", t, flags=re.IGNORECASE))
        scored.append((t, score))
    best = max(scored, key=lambda x: x[1])
    return best[0], scored

# ---------------- parsing helpers ----------------
def parse_amount_from_text(text: str) -> Optional[float]:
    patterns = [
        r"₹\s*([\d,]+(?:\.\d{1,2})?)",
        r"Rs\.?\s*([\d,]+(?:\.\d{1,2})?)",
        r"INR\s*([\d,]+(?:\.\d{1,2})?)",
        r"\b([\d,]+\.\d{1,2})\b",
        r"\b([\d,]{1,6})\b"
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            s = m.group(1).replace(",", "")
            try:
                return float(s)
            except:
                continue
    return None

def parse_upi_from_text(text: str) -> Optional[str]:
    m = re.search(r"([a-zA-Z0-9\.\-_]{3,}@[a-zA-Z0-9\.\-_]{2,})", text)
    return m.group(1) if m else None

def find_keywords_in_text(text: str) -> List[str]:
    keys = ["paid", "payment successful", "paid successfully", "successful", "txn", "transaction id", "paid to"]
    return [k for k in keys if k in text.lower()]

# ---------------- Groq verification ----------------
def call_groq_verifier(ocr_text: str, expected_amount: Optional[float], expected_upi: Optional[str]) -> Optional[dict]:
    if not GROQ_API_KEY:
        return None
    system = ("You are a payment verification assistant. Given OCR text, expected_amount (float|null), expected_upi (string|null), "
              "return ONLY a JSON object with keys: decision (accept|manual_review|reject), confidence (0-1), amount_found (number|null), upi_found (string|null), reason (string).")
    user_prompt = ("OCR_TEXT:\n" + (ocr_text[:5000] if ocr_text else "") + "\n\n"
                   f"EXPECTED_AMOUNT: {expected_amount}\nEXPECTED_UPI: {expected_upi}\n\nReturn ONLY JSON.")
    payload = {"model": GROQ_MODEL, "messages":[{"role":"system","content":system},{"role":"user","content":user_prompt}], "temperature":0.0, "max_tokens":300}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return None
        start = content.find("{"); end = content.rfind("}")
        if start == -1 or end == -1:
            return None
        jt = content[start:end+1]
        parsed = json.loads(jt)
        if parsed.get("decision") not in ("accept","manual_review","reject"):
            return None
        return parsed
    except Exception:
        logger.exception("Groq call failed")
        return None

# ---------------- local fallback verification ----------------
def local_verify(ocr_text: str, expected_amount: Optional[float], expected_upi: Optional[str]) -> dict:
    amt = parse_amount_from_text(ocr_text)
    upi = parse_upi_from_text(ocr_text)
    kws = find_keywords_in_text(ocr_text)
    notes = []
    score = 0
    if expected_amount is not None and amt is not None:
        if abs(amt - expected_amount) <= AMOUNT_TOLERANCE:
            score += 3
        else:
            notes.append(f"amount_mismatch(found={amt})")
    elif expected_amount is not None and amt is None:
        notes.append("amount_not_found")
    if expected_upi and upi:
        if upi.lower() == expected_upi.lower():
            score += 3
        else:
            notes.append(f"upi_mismatch(found={upi})")
    elif expected_upi and not upi:
        notes.append("upi_not_found")
    if kws:
        score += 2
    decision = "reject"
    if score >= 6:
        decision = "accept"
    elif score >= 3:
        decision = "manual_review"
    return {"decision":decision,"score":score,"amount_found":amt,"upi_found":upi,"notes":notes,"ocr_text":ocr_text[:2000]}

# ---------------- helpers ----------------
def welcome_text(name: Optional[str]) -> str:
    if name:
        return f"🍽️ Hey {name}! Welcome to JaiSai Restaurant  😋"
    return "🍽️ Welcome to JaiSai Restaurant ! What's your name? 😄"

def item_reply(item: str) -> str:
    replies = [
        f"Nice pick — {item} is a crowd favourite! 🔥",
        f"Yum! {item} coming right up (when you confirm 😉)",
        f"Great choice — {item} will make you smile 😄"
    ]
    return replies[hash(item) % len(replies)]

def validate_indian_phone(number: str) -> Optional[str]:
    n = number.strip().replace(" ", "").replace("-", "")
    if n.startswith("0") and len(n) == 11:
        n = n[1:]
    if n.startswith("+91") and len(n) == 13:
        candidate = n[3:]
    elif n.startswith("91") and len(n) == 12:
        candidate = n[2:]
    elif len(n) == 10 and n.isdigit():
        candidate = n
    else:
        return None
    if candidate and re.match(r"^[6-9]\d{9}$", candidate):
        return candidate
    return None

def paginate(items: List[str], page: int) -> Tuple[List[str], int]:
    pages = max(1, (len(items) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    page = max(1, min(page, pages))
    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    return items[start:end], pages

def suggest_items(text: str, max_suggestions: int = 6) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    scores = {}
    for item in MENU.keys():
        name_lower = item.lower()
        score = 0
        for t in tokens:
            if t in name_lower:
                score += 2
            if re.search(r"\b" + re.escape(t), name_lower):
                score += 1
        if score > 0:
            scores[item] = score
    if not scores:
        words = text.split()
        candidates = set()
        for w in words:
            matches = difflib.get_close_matches(w, MENU.keys(), n=max_suggestions, cutoff=0.6)
            candidates.update(matches)
        return list(candidates)[:max_suggestions]
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [i[0] for i in sorted_items][:max_suggestions]

# ---------------- UI: banner & specials ----------------
async def show_banner_and_specials(update: Update):
    caption = "🍽️ *JaiSai Restaurant *\nHere are today's specials — tap to add or scroll categories below."
    try:
        await update.message.reply_photo(photo=open(BANNER_IMAGE_PATH, "rb"), caption=caption, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(caption, parse_mode="Markdown")
    specials = CATEGORIES.get("Specials", [])
    text = "✨ *Today's Specials* ✨\n\n"
    for s in specials:
        text += f"• {s} — ₹{MENU.get(s,'N/A')}\n"
    btns = [[s] for s in specials]
    btns.append(["🔙 Categories"])
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=ReplyKeyboardMarkup(btns, resize_keyboard=True))

# ---------------- handlers ----------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    context.user_data["cart"] = []
    context.user_data["step"] = "ask_name"
    await update.message.reply_text(welcome_text(None))

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Flow: Say Hi → Banner & Specials → Categories → choose item → qty → continue/payment → phone → UPI → screenshot.")

async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("Order cancelled. Type /start to order again.")

# map emoji to ints
EMOJI_QTY = {"1️⃣": 1, "2️⃣": 2, "3️⃣": 3, "4️⃣": 4, "5️⃣": 5}

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return
    lower = text.lower()
    step = context.user_data.get("step", "")

    # if waiting screenshot, keep reminder
    if step == "wait_screenshot":
        await update.message.reply_text("📸 Waiting for your payment screenshot — upload it here.")
        return

    # ASK NAME
    if step == "ask_name":
        context.user_data["name"] = text
        context.user_data["cart"] = []
        # now ask table (1..10)
        table_buttons = [[str(i)] for i in range(1, 6)] + [[str(i)] for i in range(6, 11)]
        await update.message.reply_text(f"Thanks {text}! Which table are you at? (1-10)", reply_markup=ReplyKeyboardMarkup(table_buttons, resize_keyboard=True))
        context.user_data["step"] = "ask_table"
        return

    # ASK TABLE
    if step == "ask_table":
        if text.isdigit() and 1 <= int(text) <= 10:
            context.user_data["table"] = int(text)
            await update.message.reply_text(f"Table {text} noted. Here's our menu:")
            await show_banner_and_specials(update)
            cat_buttons = [[c] for c in CATEGORIES.keys()]
            cat_buttons.append(["View All Items"]); cat_buttons.append(["Search 🔎"])
            await update.message.reply_text("Tap a category or use Search 🔎.", reply_markup=ReplyKeyboardMarkup(cat_buttons, resize_keyboard=True))
            context.user_data["step"] = "select_item"
            return
        else:
            await update.message.reply_text("Please choose a table between 1 and 10 (tap a button).")
            return

    # greetings or auto menu triggers when not in flow
    triggers = {"hi", "hello", "hey", "menu", "food", "order", "start", "items", "specials", "today"}
    if lower in triggers and step == "":
        # ask name first
        context.user_data["step"] = "ask_name"
        await update.message.reply_text(welcome_text(None))
        return

    # search flow
    if lower in ("search", "search 🔎"):
        await update.message.reply_text("🔎 Send a keyword to search (e.g., 'biryani', 'paneer', 'chicken').")
        context.user_data["step"] = "search"
        return
    if step == "search":
        suggestions = suggest_items(text, max_suggestions=8)
        if not suggestions:
            await update.message.reply_text("No items found for that search. Try another keyword.")
            return
        btns = [[s] for s in suggestions]
        btns.append(["🔙 Categories"])
        await update.message.reply_text("I found these items — tap to choose:", reply_markup=ReplyKeyboardMarkup(btns, resize_keyboard=True))
        context.user_data["step"] = "select_item"
        return

    # category selection
    if text in CATEGORIES:
        context.user_data["current_category"] = text
        context.user_data["cat_page"] = 1
        await show_category(update, text, 1)
        context.user_data["step"] = "select_item"
        return

    # view all items
    if text == "View All Items":
        all_items = list(MENU.keys())
        context.user_data["virtual_list"] = all_items
        context.user_data["virtual_page"] = 1
        await show_virtual_page(update, all_items, 1)
        context.user_data["step"] = "select_item"
        return

    # pagination/back (works for virtual or category)
    if text in ("Next ▶️", "◀️ Back", "🔙 Categories"):
        # virtual list
        if context.user_data.get("virtual_list"):
            vlist = context.user_data["virtual_list"]
            vpage = context.user_data.get("virtual_page", 1)
            total_pages = max(1, (len(vlist) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
            if text == "Next ▶️" and vpage < total_pages:
                vpage += 1
            if text == "◀️ Back" and vpage > 1:
                vpage -= 1
            if text == "🔙 Categories":
                await show_banner_and_specials(update)
                cat_buttons = [[c] for c in CATEGORIES.keys()]
                cat_buttons.append(["View All Items"]); cat_buttons.append(["Search 🔎"])
                await update.message.reply_text("Tap a category or use Search 🔎.", reply_markup=ReplyKeyboardMarkup(cat_buttons, resize_keyboard=True))
                context.user_data.pop("virtual_list", None)
                context.user_data["step"] = "select_item"
                return
            context.user_data["virtual_page"] = vpage
            await show_virtual_page(update, vlist, vpage)
            return

        # category pages
        cat = context.user_data.get("current_category")
        if not cat:
            await show_banner_and_specials(update)
            return
        page = context.user_data.get("cat_page", 1)
        items = CATEGORIES.get(cat, [])
        pages = max(1, (len(items) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        if text == "Next ▶️" and page < pages:
            page += 1
        if text == "◀️ Back" and page > 1:
            page -= 1
        if text == "🔙 Categories":
            await show_banner_and_specials(update)
            cat_buttons = [[c] for c in CATEGORIES.keys()]
            cat_buttons.append(["View All Items"]); cat_buttons.append(["Search 🔎"])
            await update.message.reply_text("Tap a category or use Search 🔎.", reply_markup=ReplyKeyboardMarkup(cat_buttons, resize_keyboard=True))
            context.user_data["step"] = "select_item"
            return
        context.user_data["cat_page"] = page
        await show_category(update, cat, page)
        return

    # SELECT ITEM
    if context.user_data.get("step") == "select_item":
        chosen = text.split(" - ")[0].strip()
        # case-insensitive match
        if chosen not in MENU:
            cand = [k for k in MENU.keys() if k.lower() == chosen.lower()]
            if cand:
                chosen = cand[0]
        if chosen not in MENU:
            suggestions = suggest_items(text, max_suggestions=6)
            if suggestions:
                btns = [[s] for s in suggestions]; btns.append(["🔙 Categories"])
                await update.message.reply_text("I couldn't find exact item. Did you mean:", reply_markup=ReplyKeyboardMarkup(btns, resize_keyboard=True))
                return
            await update.message.reply_text("Item not found. Tap categories or use Search 🔎.")
            await show_banner_and_specials(update)
            return
        # present quantity emoji buttons
        context.user_data["current_item"] = chosen
        qty_buttons = [["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣"], ["🔙 Categories"]]
        await update.message.reply_text(item_reply(chosen))
        await update.message.reply_text("Tap quantity:", reply_markup=ReplyKeyboardMarkup(qty_buttons, resize_keyboard=True))
        context.user_data["step"] = "choose_qty"
        return

    # QUANTITY selected via emoji
    if context.user_data.get("step") == "choose_qty":
        if text in EMOJI_QTY:
            qty = EMOJI_QTY[text]
            item = context.user_data.get("current_item")
            price = MENU[item]
            total = price * qty
            cart = context.user_data.get("cart", [])
            cart.append({"item": item, "qty": qty, "price": price, "total": total})
            context.user_data["cart"] = cart
            await update.message.reply_text(f"✅ Added {item} x{qty} — ₹{total}")
            # after adding ask Continue or Go to Payment
            buttons = [["Continue Order 🛒"], ["Go to Payment 💵"]]
            await update.message.reply_text("Do you want to continue ordering or go to payment?", reply_markup=ReplyKeyboardMarkup(buttons, resize_keyboard=True))
            context.user_data["step"] = "post_cart"
            return
        if text == "🔙 Categories":
            await show_banner_and_specials(update)
            cat_buttons = [[c] for c in CATEGORIES.keys()]; cat_buttons.append(["View All Items"]); cat_buttons.append(["Search 🔎"])
            await update.message.reply_text("Tap a category or use Search 🔎.", reply_markup=ReplyKeyboardMarkup(cat_buttons, resize_keyboard=True))
            context.user_data["step"] = "select_item"
            return
        await update.message.reply_text("Tap a quantity button (1️⃣..5️⃣) or tap 🔙 Categories.")
        return

    # POST_CART
    if context.user_data.get("step") == "post_cart":
        if text == "Continue Order 🛒":
            await show_banner_and_specials(update)
            cat_buttons = [[c] for c in CATEGORIES.keys()]; cat_buttons.append(["View All Items"]); cat_buttons.append(["Search 🔎"])
            await update.message.reply_text("Tap a category or use Search 🔎.", reply_markup=ReplyKeyboardMarkup(cat_buttons, resize_keyboard=True))
            context.user_data["step"] = "select_item"
            return
        if text == "Go to Payment 💵":
            # show cart then payment options
            await show_cart(update, context)
            pay_buttons = [["UPI"], ["Add item 🔁"], ["Search 🔎"]]
            await update.message.reply_text("Choose payment method or change your mind:", reply_markup=ReplyKeyboardMarkup(pay_buttons, resize_keyboard=True))
            context.user_data["step"] = "choose_payment"
            return
        await update.message.reply_text("Choose Continue Order 🛒 or Go to Payment 💵 using the buttons.")
        return

    # CHOOSE_PAYMENT
    if context.user_data.get("step") == "choose_payment":
        if text == "Add item 🔁":
            await show_banner_and_specials(update)
            cat_buttons = [[c] for c in CATEGORIES.keys()]; cat_buttons.append(["View All Items"]); cat_buttons.append(["Search 🔎"])
            await update.message.reply_text("Tap a category or use Search 🔎.", reply_markup=ReplyKeyboardMarkup(cat_buttons, resize_keyboard=True))
            context.user_data["step"] = "select_item"
            return
        if text == "Search 🔎":
            await update.message.reply_text("🔎 Send a keyword to search (e.g., 'biryani', 'paneer').")
            context.user_data["step"] = "search"
            return
        if text != "UPI":
            await update.message.reply_text("We accept UPI only. Choose UPI, or Add item 🔁, or Search 🔎.")
            return
        # choose UPI -> ask phone
        context.user_data["payment"] = "UPI"
        await update.message.reply_text("Please send your phone number (10-digit Indian or +91).")
        context.user_data["step"] = "ask_phone"
        return

    # ASK_PHONE
    if context.user_data.get("step") == "ask_phone":
        phone = validate_indian_phone(text)
        if not phone:
            await update.message.reply_text("Invalid phone. Send 10-digit like 9876543210 or +919876543210.")
            return
        context.user_data["phone"] = phone
        total = sum(c["total"] for c in context.user_data.get("cart", []))
        context.user_data["expected_amount"] = total
        await update.message.reply_text(f"Thanks! Sending the UPI QR. Please pay ₹{total} and upload your payment screenshot here.")
        try:
            await update.message.reply_photo(photo=open(QR_IMAGE_PATH, "rb"),
                                             caption=f"Scan to pay ₹{total}\nUPI ID: {YOUR_UPI_ID}")
        except Exception:
            await update.message.reply_text(f"(QR file: {QR_IMAGE_PATH})\nScan to pay ₹{total}\nUPI ID: {YOUR_UPI_ID}")
        context.user_data["step"] = "wait_screenshot"
        return

    # fallback: show banner & categories
    await show_banner_and_specials(update)
    cat_buttons = [[c] for c in CATEGORIES.keys()]; cat_buttons.append(["View All Items"]); cat_buttons.append(["Search 🔎"])
    await update.message.reply_text("Tap a category or use Search 🔎.", reply_markup=ReplyKeyboardMarkup(cat_buttons, resize_keyboard=True))
    context.user_data["step"] = "select_item"

# ---------------- show pages & cart ----------------
async def show_category(update: Update, category: str, page: int = 1):
    items = CATEGORIES.get(category, [])
    if not items:
        await update.message.reply_text("No items in this category.")
        return
    page_items, pages = paginate(items, page)
    text = f"🍽️ *{category}* — Page {page}/{pages}\n\n"
    for it in page_items:
        text += f"• {it} — ₹{MENU.get(it,'N/A')}\n"
    btns = [[it] for it in page_items]
    nav = []
    if page > 1: nav.append("◀️ Back")
    nav.append("🔙 Categories")
    if page < pages: nav.append("Next ▶️")
    btns.append(nav)
    try:
        await update.message.reply_photo(photo=open(BANNER_IMAGE_PATH, "rb"),
                                         caption=text, parse_mode="Markdown",
                                         reply_markup=ReplyKeyboardMarkup(btns, resize_keyboard=True))
    except Exception:
        await update.message.reply_text(text, parse_mode="Markdown",
                                        reply_markup=ReplyKeyboardMarkup(btns, resize_keyboard=True))

async def show_virtual_page(update: Update, items: List[str], page: int = 1):
    page_items, pages = paginate(items, page)
    text = f"🍽️ *All Items* — Page {page}/{pages}\n\n"
    for it in page_items:
        text += f"• {it} — ₹{MENU[it]}\n"
    btns = [[it] for it in page_items]
    nav = []
    if page > 1: nav.append("◀️ Back")
    nav.append("🔙 Categories")
    if page < pages: nav.append("Next ▶️")
    btns.append(nav)
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=ReplyKeyboardMarkup(btns, resize_keyboard=True))

async def show_cart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cart = context.user_data.get("cart", [])
    if not cart:
        await update.message.reply_text("Your cart is empty. Add items from the menu 😋")
        return
    msg = "🛒 *Your Cart:*\n\n"
    total = 0
    for c in cart:
        msg += f"{c['item']} x{c['qty']} = ₹{c['total']}\n"
        total += c['total']
    msg += f"\n*Grand Total: ₹{total}*"
    await update.message.reply_text(msg, parse_mode="Markdown")

# ---------------- photo handler (payment screenshot) ----------------
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    step = context.user_data.get("step", "")
    if step != "wait_screenshot":
        await show_banner_and_specials(update)
        await update.message.reply_text("If you intended to confirm a payment, follow the order flow. Type /start to begin.")
        return

    photo_file = await update.message.photo[-1].get_file()
    ts = int(datetime.datetime.now().timestamp())
    filename = f"payment_{update.message.from_user.id}_{ts}.jpg"
    await photo_file.download_to_drive(filename)

    best_text, scored = ocr_multi_pass(filename)
    expected_amount = context.user_data.get("expected_amount")
    groq_res = call_groq_verifier(best_text, expected_amount, YOUR_UPI_ID)
    used_groq = True
    if groq_res is None:
        used_groq = False
        groq_res = local_verify(best_text, expected_amount, YOUR_UPI_ID)

    # aggressive crop + retry if manual_review
    if groq_res.get("decision") == "manual_review":
        try:
            img = cv2.imread(filename)
            h, w = img.shape[:2]
            cy, cx = h//2, w//2
            crop = img[max(0, cy - int(h*0.35)):min(h, cy + int(h*0.35)), max(0, cx - int(w*0.4)):min(w, cx + int(w*0.4))]
            crop_path = f"/tmp/crop_{ts}.png"
            cv2.imwrite(crop_path, crop)
            extra_text, _ = ocr_multi_pass(crop_path)
            combined = best_text + "\n\n" + extra_text
            retry = call_groq_verifier(combined, expected_amount, YOUR_UPI_ID)
            if retry:
                groq_res = retry
        except Exception:
            logger.exception("Aggressive OCR failed")

    decision = groq_res.get("decision")
    if decision == "accept":
        await update.message.reply_text("🎉 Payment verified! Your order is confirmed. Enjoy your meal 🍽️")
        # optionally notify admin with screenshot and order summary
        if ADMIN_CHAT_ID:
            await context.bot.send_photo(chat_id=ADMIN_CHAT_ID, photo=open(filename, "rb"),
                                         caption=f"Order accepted. OCR excerpt:\n{best_text[:600]}\nUser data: {pformat(context.user_data)}")
        context.user_data.clear()
        return
    elif decision == "manual_review":
        await update.message.reply_text("⚠️ We couldn't fully auto-verify. Your order is saved for manual review — we'll confirm shortly.")
        if ADMIN_CHAT_ID:
            await context.bot.send_photo(chat_id=ADMIN_CHAT_ID, photo=open(filename, "rb"),
                                         caption=f"Manual review needed. OCR excerpt:\n{best_text[:600]}\nUser data: {pformat(context.user_data)}")
        context.user_data.clear()
        return
    else:
        await update.message.reply_text("❌ Payment screenshot invalid (amount/UPI mismatch). Please re-check and upload a valid screenshot showing amount and paid-to info.")
        if best_text:
            await update.message.reply_text(f"Detected text (debug):\n{best_text.strip()[:700]}")
        context.user_data["step"] = "wait_screenshot"
        return

# ---------------- commands & main ----------------
def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("Friends Catering bot running (no gsheet).")
    app.run_polling()

if __name__ == "__main__":
    main()
