import requests

url = "http://localhost:5000/predict"
products = ["Carlsberg", "Farmfresh yogurt 1L", "Pampers Diapers Size M", "Heinz Ketchup 300g"]

for product in products:
    payload = {"product": product}
    response = requests.post(url, json=payload)
    print(f"\nProduct: {product}")
    print(response.json())