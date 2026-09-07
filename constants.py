"""
Daftar kategori valid, diambil dari nilai unik UsedCarsSA_Clean_EN.csv
(dipakai sebagai opsi selectbox di Streamlit supaya user tidak mengetik
kategori yang tidak dikenali OneHotEncoder saat training).
"""

ORIGIN_OPTS = ["Saudi", "Gulf Arabic", "Other", "Unknown"]

COLOR_OPTS = [
    "Black", "White", "Silver", "Grey", "Blue", "Red", "Brown", "Golden",
    "Green", "Navy", "Bronze", "Orange", "Yellow", "Oily", "Another Color",
]

OPTIONS_OPTS = ["Standard", "Semi Full", "Full"]

FUEL_TYPE_OPTS = ["Gas", "Diesel", "Hybrid"]

GEAR_TYPE_OPTS = ["Automatic", "Manual"]

REGION_OPTS = [
    "Riyadh", "Jeddah", "Makkah", "Dammam", "Khobar", "Taef", "Qassim",
    "Al-Ahsa", "Jubail", "Yanbu", "Hail", "Tabouk", "Al-Medina", "Abha",
    "Aseer", "Najran", "Jazan", "Sabya", "Hafar Al-Batin", "Sakaka",
    "Al-Jouf", "Qurayyat", "Al-Baha", "Al-Namas", "Besha", "Arar",
    "Wadi Dawasir",
]

# Make -> daftar Type yang pernah muncul di data training untuk Make tersebut.
# Dipakai supaya dropdown "Type" otomatis nyaring sesuai Make yang dipilih.
MAKE_TYPE_MAP = {
    "Aston Martin": ["DB9", "Vanquish", "Vantage"],
    "Audi": ["A3", "A4", "A5", "A6", "A7", "A8", "Q3", "Q5", "Q7", "Q8"],
    "BMW": ["The 2", "The 3", "The 4", "The 5", "The 6", "The 7", "The X5", "The X6"],
    "BYD": ["F3", "S5"],
    "Bentley": ["Continental", "Flying Spur", "Mulsanne"],
    "Cadillac": ["ATS", "CT6", "CTS", "Escalade", "SRX", "XT5", "XTS"],
    "Changan": ["CS35", "CS35 Plus", "CS55", "CS75", "Eado"],
    "Chery": ["Arrizo", "Tiggo"],
    "Chevrolet": ["Aveo", "Camaro", "Captiva", "Cruze", "Impala", "Malibu",
                  "Optra", "Silverado", "Sonic", "Suburban", "Tahoe", "Traverse"],
    "Chrysler": ["300"],
    "Dodge": ["Challenger", "Charger", "Durango", "Journey", "Ram"],
    "FAW": ["V2"],
    "Fiat": ["Panda"],
    "Foton": ["View"],
    "GAC": ["GS4", "GS8"],
    "GMC": ["Acadia", "Sierra", "Terrain", "Yukon"],
    "Genesis": ["G70", "G80", "G90"],
    "Geely": ["Emgrand"],
    "Great Wall": ["Wingle", "H6"],
    "Honda": ["Accord", "City", "Civic", "CR-V", "Odyssey", "Pilot"],
    "Hummer": ["H2", "H3"],
    "Hyundai": ["Accent", "Azera", "Creta", "Elantra", "Genesis", "H1",
                "Santa Fe", "Sonata", "Tucson"],
    "Infiniti": ["FX", "G", "M", "Q50", "Q60", "Q70", "QX56", "QX60", "QX80"],
    "Isuzu": ["D-Max"],
    "JAC": ["S3"],
    "Jaguar": ["F-Pace", "XE", "XF", "XJ"],
    "Jeep": ["Cherokee", "Compass", "Grand Cherokee", "Renegade", "Wrangler"],
    "Kia": ["Cadenza", "Carnival", "Cerato", "Optima", "Rio", "Sorento",
            "Soul", "Sportage"],
    "Land Rover": ["Discovery", "Range Rover", "Range Rover Evoque",
                   "Range Rover Sport", "Range Rover Velar"],
    "Lexus": ["ES", "GS", "GX", "IS", "LS", "LX", "NX", "RX"],
    "Lincoln": ["MKS", "MKX", "MKZ", "Navigator"],
    "Lotus": ["Elise"],
    "MG": ["MG5", "MG6", "RX5", "ZS"],
    "Maserati": ["Ghibli", "Quattroporte"],
    "Maxus": ["G50", "T60"],
    "Mazda": ["2", "3", "6", "CX-3", "CX-5", "CX-9"],
    "Mercedes": ["C 200", "C 300", "CLA 250", "E 200", "E 300", "GLA 250",
                 "GLC 300", "GLE 350", "GLS 450", "S 500"],
    "Mitsubishi": ["ASX", "Attrage", "Eclipse Cross", "Lancer", "Mirage",
                   "Outlander", "Pajero"],
    "Nissan": ["Altima", "Armada", "Maxima", "Micra", "Murano", "Patrol",
               "Pathfinder", "Sentra", "Sunny", "Urvan", "X-Trail"],
    "Opel": ["Astra", "Corsa"],
    "Peugeot": ["2008", "208", "301", "3008", "307", "308", "5008", "508"],
    "Porsche": ["911", "Cayenne", "Macan", "Panamera"],
    "Renault": ["Duster", "Fluence", "Koleos", "Symbol"],
    "Rolls Royce": ["Ghost", "Phantom", "Wraith"],
    "Skoda": ["Octavia", "Superb"],
    "Speranza": ["A516"],
    "Subaru": ["Forester", "Impreza", "Legacy", "Outback", "XV"],
    "Suzuki": ["Celerio", "Ciaz", "Dzire", "Ertiga", "Swift", "Vitara"],
    "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "Fortuner", "Hilux",
               "Land Cruiser", "Prado", "RAV4", "Yaris"],
    "Volkswagen": ["CC", "Golf", "Jetta", "Passat", "Tiguan", "Touareg"],
    "Volvo": ["S60", "S80", "S90", "XC60", "XC90"],
}

MAKE_OPTS = sorted(MAKE_TYPE_MAP.keys())

# Fallback kalau Type yang diinginkan belum ke-cover di MAKE_TYPE_MAP
GENERIC_TYPE_FALLBACK = "Lainnya (ketik manual)"
